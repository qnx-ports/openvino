#include <sched.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <sys/syspage.h>
#include <sys/neutrino.h>

#include "dev/threading/parallel_custom_arena.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

namespace ov {
#define SOCKET_MASK 0xFFFF0000

CPU::CPU() {
    std::set<uint8_t> physical_cores;
    std::map<std::string, int> system_info_table = {
        {"logical_cores", 0},
        {"physical_cores", 0},
        {"sockets", 1},
        {"numa_nodes", 1},
        {"blocked_cores", 0},
        {"p_cores", 0},
        {"e_cores", 0}
    };

    uint16_t num_cores = _syspage_ptr->num_cpu;
    struct cpuinfo_entry *cpus = SYSPAGE_ENTRY(cpuinfo);
    uint32_t runmask = 0;
    uint32_t max_speed = 0;

    // find physical cores
    for (uint16_t idx = 0; idx < num_cores; idx++ ) {
        uint8_t hwid = static_cast<uint8_t>(cpus[idx].smp_hwcoreid);
        uint32_t sock_id = cpus[idx].smp_hwcoreid & SOCKET_MASK;
        physical_cores.insert(hwid);
        max_speed = std::max(max_speed, cpus[idx].speed);
    }

    // find blocked cores
    if (ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET, (void *)&runmask) == -1) {
        OPENVINO_THROW("Cannot find CPUs which allow openvino execution.");
    }

    // find P and E cores
    uint16_t pcount = 0;
    for (uint16_t idx = 0; idx < num_cores; idx++ ) {
        if(cpus[idx].speed == max_speed)
            pcount++;
    }

    system_info_table["logical_cores"] = static_cast<int32_t>(num_cores);
    system_info_table["physical_cores"] = physical_cores.size();
    system_info_table["blocked_cores"] = system_info_table["logical_cores"] - __builtin_popcount(runmask);
    if (pcount == system_info_table["physical_cores"]) {
        system_info_table["p_cores"] = 0;
        system_info_table["e_cores"] = pcount;
    }
    else {
        system_info_table["p_cores"] = pcount;
        system_info_table["e_cores"] = system_info_table["physical_cores"] - pcount;
    }

    parse_processor_info_qnx(system_info_table, _processors, _numa_nodes, _sockets, _cores,
                             _blocked_cores, _proc_type_table, _cpu_mapping_table);
    _org_proc_type_table = _proc_type_table;
    _numaid_mapping_table = {{0, 0}};
    _socketid_mapping_table = {{0, 0}};

    cpu_debug();
}

void parse_processor_info_qnx(std::map<std::string, int> system_info_table,
                              int& _processors,
                              int& _numa_nodes,
                              int& _sockets,
                              int& _cores,
                              int& _blocked_cores,
                              std::vector<std::vector<int>>& _proc_type_table,
                              std::vector<std::vector<int>>& _cpu_mapping_table) {

    _processors = system_info_table["logical_cores"];
    _numa_nodes = 1;
    _sockets = 1;
    _cores = system_info_table["physical_cores"];
    _blocked_cores = system_info_table["blocked_cores"];

    _proc_type_table.resize(1, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));

    _proc_type_table[0][ALL_PROC] = _processors;
    _proc_type_table[0][MAIN_CORE_PROC] = system_info_table["p_cores"];
    _proc_type_table[0][HYPER_THREADING_PROC] = _processors - _cores;
    _proc_type_table[0][EFFICIENT_CORE_PROC] = system_info_table["e_cores"];
    _proc_type_table[0][LP_EFFICIENT_CORE_PROC] = 0;
    _proc_type_table[0][PROC_NUMA_NODE_ID] = -1;
    _proc_type_table[0][PROC_SOCKET_ID] = -1;

    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));
    struct cpuinfo_entry *cpus = SYSPAGE_ENTRY(cpuinfo);
    struct cacheattr_entry *caches = SYSPAGE_ENTRY(cacheattr);
    uint32_t runmask = 0;
    std::map<int, std::vector<int>> l2_cache_map;

    if (ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET, (void *)&runmask) == -1) {
        OPENVINO_THROW("Cannot find CPUs which allow openvino execution.");
    }

    if (system_info_table["p_cores"] == 0) {
        for (int idx = 0; idx < _processors; idx++ ) {
            _cpu_mapping_table[idx][CPU_MAP_PROCESSOR_ID] = idx;
            _cpu_mapping_table[idx][CPU_MAP_NUMA_NODE_ID] = 0;
            _cpu_mapping_table[idx][CPU_MAP_SOCKET_ID] = 0;
            _cpu_mapping_table[idx][CPU_MAP_CORE_ID] = idx;
            _cpu_mapping_table[idx][CPU_MAP_CORE_TYPE] = 2;

            uint8_t data_cache = cpus[idx].data_cache;
            uint32_t l2_cache_idx = caches[data_cache].next;
            l2_cache_map[l2_cache_idx].push_back(idx);
            _cpu_mapping_table[idx][CPU_MAP_USED_FLAG] = (runmask & (1 << idx)) ? NOT_USED : CPU_BLOCKED;
        }
    }
    else {
        int e_core_idx = 0;
        for (int idx = 0; idx < _processors; idx++ ) {
            _cpu_mapping_table[idx][CPU_MAP_PROCESSOR_ID] = idx;
            _cpu_mapping_table[idx][CPU_MAP_NUMA_NODE_ID] = 0;
            _cpu_mapping_table[idx][CPU_MAP_SOCKET_ID] = 0;

            // assuming sequential layout
            if (idx < 2 * system_info_table["p_cores"]) {
                _cpu_mapping_table[idx][CPU_MAP_CORE_ID] = (idx & 1) ? idx - 1 : idx;
                _cpu_mapping_table[idx][CPU_MAP_CORE_TYPE] = (idx & 1) ? 4 : 1;
            }
            else {
                if (e_core_idx == 0)
                  e_core_idx = idx / 2;
                _cpu_mapping_table[idx][CPU_MAP_CORE_ID] = e_core_idx;
                _cpu_mapping_table[idx][CPU_MAP_CORE_TYPE] = 2;
                e_core_idx++;
            }

            uint8_t data_cache = cpus[idx].data_cache;
            uint32_t l2_cache_idx = caches[data_cache].next;
            l2_cache_map[l2_cache_idx].push_back(idx);
            _cpu_mapping_table[idx][CPU_MAP_USED_FLAG] = (runmask & (1 << idx)) ? NOT_USED : CPU_BLOCKED;
        }
    }

    uint32_t l2_cache_idx = 0;
    for(auto &l2_cache : l2_cache_map) {
        for(auto &idx : l2_cache.second)
            _cpu_mapping_table[idx][CPU_MAP_GROUP_ID] = l2_cache_idx;
        l2_cache_idx;
    }

}
}

