// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dev/threading/thread_affinity.hpp"

#include <cerrno>
#include <climits>
#include <tuple>
#include <utility>

#include "openvino/runtime/system_conf.hpp"

#if !(defined(__APPLE__) || defined(__EMSCRIPTEN__) || defined(_WIN32))
#    include <sched.h>
#    include <unistd.h>
#endif

#if defined(__QNX__)
#    include <sys/neutrino.h>
#    include <sys/syspage.h>
#endif

namespace ov {
namespace threading {
#if !(defined(__APPLE__) || defined(__EMSCRIPTEN__) || defined(_WIN32) || defined(__QNX__))
std::tuple<CpuSet, int> get_process_mask() {
    for (int ncpus = sizeof(cpu_set_t) / CHAR_BIT; ncpus < 32768 /* reasonable limit of #cores*/; ncpus <<= 1) {
        CpuSet mask{CPU_ALLOC(ncpus)};
        if (nullptr == mask)
            break;
        const size_t size = CPU_ALLOC_SIZE(ncpus);
        CPU_ZERO_S(size, mask.get());
        // the result fits the mask
        if (0 == sched_getaffinity(getpid(), size, mask.get())) {
            return std::make_tuple(std::move(mask), ncpus);
        }
        // other error
        if (errno != EINVAL)
            break;
    }
    return std::make_tuple(nullptr, 0);
}

/* Release the cores affinity mask for the current process */
void release_process_mask(cpu_set_t* mask) {
    if (nullptr != mask)
        CPU_FREE(mask);
}

bool pin_current_thread_by_mask(int ncores, const CpuSet& procMask) {
    return 0 == sched_setaffinity(0, CPU_ALLOC_SIZE(ncores), procMask.get());
}

bool pin_thread_to_vacant_core(int thrIdx,
                               int hyperthreads,
                               int ncores,
                               const CpuSet& procMask,
                               const std::vector<int>& cpu_ids) {
    if (procMask == nullptr)
        return false;
    const size_t size = CPU_ALLOC_SIZE(ncores);
    const int num_cpus = CPU_COUNT_S(size, procMask.get());
    thrIdx %= num_cpus;  // To limit unique number in [; num_cpus-1] range

    int mapped_idx;
    if (cpu_ids.size() > 0) {
        mapped_idx = cpu_ids[thrIdx];
    } else {
        // Place threads with specified step
        int cpu_idx = 0;
        for (int i = 0, offset = 0; i < thrIdx; ++i) {
            cpu_idx += hyperthreads;
            if (cpu_idx >= num_cpus)
                cpu_idx = ++offset;
        }

        // Find index of 'cpu_idx'-th bit that equals to 1
        mapped_idx = -1;
        while (cpu_idx >= 0) {
            mapped_idx++;
            if (CPU_ISSET_S(mapped_idx, size, procMask.get()))
                --cpu_idx;
        }
    }

    CpuSet targetMask{CPU_ALLOC(ncores)};
    CPU_ZERO_S(size, targetMask.get());
    CPU_SET_S(mapped_idx, size, targetMask.get());
    bool res = pin_current_thread_by_mask(ncores, targetMask);
    return res;
}

bool pin_current_thread_to_socket(int socket) {
    auto proc_type_table = get_org_proc_type_table();
    const int sockets = proc_type_table.size() > 1 ? proc_type_table.size() - 1 : 1;
    const int cores = proc_type_table[0][MAIN_CORE_PROC];
    const int cores_per_socket = cores / sockets;

    int ncpus = 0;
    CpuSet mask;
    std::tie(mask, ncpus) = get_process_mask();
    CpuSet targetMask{CPU_ALLOC(ncpus)};
    const size_t size = CPU_ALLOC_SIZE(ncpus);
    CPU_ZERO_S(size, targetMask.get());

    for (int core = socket * cores_per_socket; core < (socket + 1) * cores_per_socket; core++) {
        CPU_SET_S(core, size, targetMask.get());
    }
    // respect the user-defined mask for the entire process
    CPU_AND_S(size, targetMask.get(), targetMask.get(), mask.get());
    bool res = false;
    if (CPU_COUNT_S(size, targetMask.get())) {  //  if we have non-zero mask to set
        res = pin_current_thread_by_mask(ncpus, targetMask);
    }
    return res;
}
#elif defined(_WIN32)
std::tuple<CpuSet, int> get_process_mask() {
    DWORD_PTR pro_mask, sys_mask;
    if (0 != GetProcessAffinityMask(GetCurrentProcess(), &pro_mask, &sys_mask)) {
        CpuSet mask = std::make_unique<cpu_set_t>(pro_mask);
        return std::make_tuple(std::move(mask), 0);
    }
    return std::make_tuple(nullptr, 0);
}
void release_process_mask(cpu_set_t*) {}

bool pin_thread_to_vacant_core(int thrIdx,
                               int hyperthreads,
                               int ncores,
                               const CpuSet& procMask,
                               const std::vector<int>& cpu_ids) {
    auto proc_type_table = get_proc_type_table();
    if (proc_type_table.size() > 1) {
        int cores_in_numa = proc_type_table[1][MAIN_CORE_PROC] + proc_type_table[1][HYPER_THREADING_PROC];
        GROUP_AFFINITY group;
        group.Group = get_numa_node_id(cpu_ids[thrIdx]);
        group.Mask = DWORD_PTR(1) << (cpu_ids[thrIdx] % cores_in_numa);
        group.Reserved[0] = 0;
        group.Reserved[1] = 0;
        group.Reserved[2] = 0;
        return 0 != SetThreadGroupAffinity(GetCurrentThread(), &group, NULL);
    } else {
        return 0 != SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu_ids[thrIdx]);
    }
}
bool pin_current_thread_by_mask(int ncores, const CpuSet& procMask) {
    DWORD_PTR mask = *procMask.get();
    return 0 != SetThreadAffinityMask(GetCurrentThread(), mask);
}
bool pin_current_thread_to_socket(int socket) {
    return false;
}
#elif __QNX__
std::tuple<CpuSet, int> get_process_mask() {
    uint32_t *runmask = new uint32_t(0);
    if (ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET, reinterpret_cast<void*>(runmask)) != -1) {
        CpuSet mask{runmask};
        return std::make_tuple(std::move(mask), static_cast<int>(sizeof(uint32_t)));
    }
    delete runmask;
    return std::make_tuple(CpuSet(nullptr), 0);
}

void release_process_mask(cpu_set_t* mask) {
    if (mask != nullptr) {
        delete mask;
    }
}

bool pin_current_thread_by_mask(int ncores, const CpuSet& procMask) {
    if (procMask == nullptr) {
        return false;
    }
    uint32_t mask = *procMask.get();
    return ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET, reinterpret_cast<void*>(&mask)) != -1;
}

bool pin_thread_to_vacant_core(int thrIdx,
                               int hyperthreads,
                               int ncores,
                               const CpuSet& procMask,
                               const std::vector<int>& cpu_ids) {
    if (procMask == nullptr) {
        return false;
    }
    const uint32_t mask_val = *procMask.get();
    const int num_cpus = __builtin_popcount(mask_val);
    if (num_cpus == 0) {
        return false;
    }
    thrIdx %= num_cpus;

    int mapped_idx = -1;
    if (cpu_ids.size() > 0) {
        mapped_idx = cpu_ids[thrIdx];
    } else {
        int cpu_idx = 0;
        for (int i = 0, offset = 0; i < thrIdx; ++i) {
            cpu_idx += hyperthreads;
            if (cpu_idx >= num_cpus) {
                cpu_idx = ++offset;
            }
        }
        while (cpu_idx >= 0) {
            mapped_idx++;
            if (mask_val & (1U << mapped_idx)) {
                --cpu_idx;
            }
        }
    }

    CpuSet targetMask{new uint32_t(1U << mapped_idx)};
    return pin_current_thread_by_mask(ncores, targetMask);
}
#else   // no threads pinning/binding on MacOS
std::tuple<CpuSet, int> get_process_mask() {
    return std::make_tuple(nullptr, 0);
}
void release_process_mask(cpu_set_t*) {}

bool pin_thread_to_vacant_core(int thrIdx,
                               int hyperthreads,
                               int ncores,
                               const CpuSet& procMask,
                               const std::vector<int>& cpu_ids) {
    return false;
}
bool pin_current_thread_by_mask(int ncores, const CpuSet& procMask) {
    return false;
}
bool pin_current_thread_to_socket(int socket) {
    return false;
}
#endif  // !(defined(__APPLE__) || defined(__EMSCRIPTEN__) || defined(_WIN32))
}  // namespace threading
}  // namespace ov
