import threading
import psutil
import os
import platform # Added platform import
from typing import Dict, Optional, List, Tuple
from enum import Enum, auto

class ResourceType(Enum):
    CPU = auto()
    GPU = auto() # Generic GPU, could be CUDA, ROCm, etc.
    ANE = auto() # Apple Neural Engine
    MPS = auto() # Apple Metal Performance Shaders

class ResourceManager:
    """
    Manages hardware resources available to the compiler and runtime.
    Tracks total resources and current allocations.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._resources: Dict[ResourceType, int] = {}
        self._allocations: Dict[str, Dict[ResourceType, int]] = {} # task_id -> {resource: count}
        self._detailed_cpu_info: Dict[str, Any] = {}
        self._detailed_gpu_info: List[Dict[str, Any]] = []

        self._detect_all_resources()
        self._initialized = True
        # print(f"ResourceManager initialized with: {self._resources}")


    def _detect_all_resources(self):
        """Detects available hardware resources."""
        self._resources[ResourceType.CPU] = os.cpu_count() or 1
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            self._detailed_cpu_info = {
                'brand': info.get('brand_raw', 'Unknown'),
                'arch': info.get('arch_string_raw', platform.machine()),
                'hz': info.get('hz_advertised_friendly', 'Unknown'),
                'flags': info.get('flags', [])
            }
        except ImportError:
            self._detailed_cpu_info = {'arch': platform.machine()}


        # GPU detection (simplified, focusing on PyTorch for CUDA/MPS)
        gpus = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpus = torch.cuda.device_count()
                for i in range(gpus):
                    props = torch.cuda.get_device_properties(i)
                    self._detailed_gpu_info.append({
                        'name': props.name,
                        'type': 'cuda',
                        'memory_mb': props.total_memory // (1024*1024)
                    })
            elif torch.backends.mps.is_available() and platform.system() == "Darwin" and platform.machine() == "arm64":
                self._resources[ResourceType.MPS] = 1 # MPS is a single logical device for PyTorch
                self._detailed_gpu_info.append({
                    'name': 'Apple MPS',
                    'type': 'mps',
                    'memory_mb': -1 # Difficult to query precise MPS memory limit directly
                })
        except ImportError:
            pass # PyTorch not available
        self._resources[ResourceType.GPU] = gpus


        # ANE detection (conceptual, as direct user-mode access is limited)
        if platform.system() == "Darwin" and platform.machine() == "arm64":
             # ANE is implicitly used by CoreML. For our compiler, its "availability" means we can target it.
            self._resources[ResourceType.ANE] = 1
        else:
            self._resources[ResourceType.ANE] = 0


    def get_total_resources(self) -> Dict[ResourceType, int]:
        """Returns the total amount of each detected resource."""
        with self._lock:
            return self._resources.copy()

    def get_detailed_cpu_info(self) -> Dict[str, Any]:
        return self._detailed_cpu_info.copy()

    def get_detailed_gpu_info(self) -> List[Dict[str, Any]]:
        return self._detailed_gpu_info[:]

    def allocate_resources(self, task_id: str, requested_resources: Dict[ResourceType, int]) -> bool:
        """
        Attempts to allocate the requested resources for a task.
        Returns True if successful, False otherwise.
        """
        with self._lock:
            # Check availability
            for resource_type, count in requested_resources.items():
                if self._resources.get(resource_type, 0) < count:
                    # print(f"Failed to allocate {count} of {resource_type.name} for {task_id}. Available: {self._resources.get(resource_type,0)}")
                    return False
            
            # Allocate
            for resource_type, count in requested_resources.items():
                self._resources[resource_type] -= count
            
            self._allocations[task_id] = requested_resources
            # print(f"Allocated {requested_resources} to {task_id}. Remaining: {self._resources}")
            return True

    def release_resources(self, task_id: str) -> None:
        """Releases resources allocated to a specific task."""
        with self._lock:
            if task_id in self._allocations:
                released_resources = self._allocations.pop(task_id)
                for resource_type, count in released_resources.items():
                    self._resources[resource_type] += count
                # print(f"Released {released_resources} from {task_id}. Remaining: {self._resources}")
            else:
                # print(f"Warning: Attempted to release resources for unknown task_id: {task_id}")
                pass

    def get_available_resources(self) -> Dict[ResourceType, int]:
        """Returns the currently available (unallocated) resources."""
        with self._lock:
            return self._resources.copy()

# Example Usage (for direct testing)
if __name__ == "__main__":
    from typing import Any # Added for example
    manager = ResourceManager()
    print("Initial Resources:", manager.get_available_resources())
    print("CPU Info:", manager.get_detailed_cpu_info())
    print("GPU Info:", manager.get_detailed_gpu_info())

    req1 = {ResourceType.CPU: 2, ResourceType.ANE: 1}
    if manager.allocate_resources("task_model_training", req1):
        print("task_model_training allocated:", req1)
        print("Available after task1:", manager.get_available_resources())
    else:
        print("Failed to allocate for task_model_training")

    req2 = {ResourceType.CPU: os.cpu_count()} # try to take all cpus
    if manager.allocate_resources("task_data_processing", req2):
        print("task_data_processing allocated:", req2)
        print("Available after task2:", manager.get_available_resources())
    else:
        print(f"Failed to allocate for task_data_processing, requested: {req2}, available {manager.get_available_resources()}")

    manager.release_resources("task_model_training")
    print("Available after releasing task1:", manager.get_available_resources())
    
    manager.release_resources("task_data_processing")
    print("Available after releasing task2:", manager.get_available_resources()) 