from queue import PriorityQueue
from typing import Callable, Any, Optional, Dict, Tuple
from dataclasses import dataclass, field
import threading
import time
import uuid

from .resource_manager import ResourceManager, ResourceType # Use relative import

@dataclass(order=True)
class Task:
    priority: int # Lower number means higher priority
    task_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    name: str = field(compare=False, default="Unnamed Task")
    requested_resources: Dict[ResourceType, int] = field(compare=False, default_factory=dict)
    func: Callable[..., Any] = field(compare=False)
    args: Tuple = field(compare=False, default_factory=tuple)
    kwargs: Dict[str, Any] = field(compare=False, default_factory=dict)
    # Additional metadata
    submission_time: float = field(compare=False, default_factory=time.time)
    deadline: Optional[float] = field(compare=False, default=None) # Optional deadline for the task
    dependencies: list[str] = field(compare=False, default_factory=list) # List of task_ids this task depends on

class TaskScheduler:
    """
    Manages a queue of tasks, allocating resources and executing them based on priority
    and resource availability. This is a simplified scheduler.
    """
    def __init__(self, resource_manager: Optional[ResourceManager] = None, max_workers: int = 4):
        self._task_queue = PriorityQueue()
        self._resource_manager = resource_manager or ResourceManager() # Use singleton instance by default
        self._task_status: Dict[str, str] = {} # task_id -> status ("pending", "running", "completed", "failed")
        self._task_results: Dict[str, Any] = {}
        self._completed_dependencies: Dict[str, set[str]] = {} # task_id -> {completed_dependency_ids}

        self._lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._max_workers = max_workers
        self._stop_event = threading.Event()

        self._start_workers()

    def _start_workers(self):
        for i in range(self._max_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"SchedulerWorker-{i}", daemon=True)
            self._workers.append(worker)
            worker.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            task_to_run: Optional[Task] = None
            with self._lock:
                if not self._task_queue.empty():
                    # Check head of queue without removing, to see if dependencies are met
                    # This is a simplification. A real scheduler might have separate queues or more complex logic.
                    potential_task = self._task_queue.queue[0] # Peek
                    if self._are_dependencies_met(potential_task):
                        if self._resource_manager.allocate_resources(potential_task.task_id, potential_task.requested_resources):
                            task_to_run = self._task_queue.get() # Actually remove it now
                            self._task_status[task_to_run.task_id] = "running"
                        # else: # Resources not available, task remains in queue
                    # else: # Dependencies not met, task remains in queue
            
            if task_to_run:
                try:
                    # print(f"Worker {threading.current_thread().name} starting task {task_to_run.name} ({task_to_run.task_id})")
                    result = task_to_run.func(*task_to_run.args, **task_to_run.kwargs)
                    with self._lock:
                        self._task_status[task_to_run.task_id] = "completed"
                        self._task_results[task_to_run.task_id] = result
                        self._notify_dependents(task_to_run.task_id)
                    # print(f"Worker {threading.current_thread().name} completed task {task_to_run.name}")
                except Exception as e:
                    with self._lock:
                        self._task_status[task_to_run.task_id] = "failed"
                        self._task_results[task_to_run.task_id] = e # Store exception as result
                        self._notify_dependents(task_to_run.task_id) # Dependents might still run or handle failure
                    # print(f"Worker {threading.current_thread().name} failed task {task_to_run.name}: {e}")
                finally:
                    self._resource_manager.release_resources(task_to_run.task_id)
            else:
                time.sleep(0.1) # Wait if no task was runnable

    def _are_dependencies_met(self, task: Task) -> bool:
        if not task.dependencies:
            return True
        completed_deps = self._completed_dependencies.get(task.task_id, set())
        for dep_id in task.dependencies:
            dep_status = self._task_status.get(dep_id)
            if dep_status == "completed":
                completed_deps.add(dep_id)
            elif dep_status == "failed": # If a dependency failed, this task might be unrunnable or handle it
                completed_deps.add(dep_id) # Mark as 'done' for dependency check purpose
                # Potentially add logic here to mark task as failed_dependency
            elif dep_id not in completed_deps:
                return False
        self._completed_dependencies[task.task_id] = completed_deps
        return len(completed_deps) == len(task.dependencies)

    def _notify_dependents(self, completed_task_id: str):
        # This is a simplification. A more robust system would track reverse dependencies.
        # For now, dependents will re-check their dependencies in the worker loop.
        pass

    def submit_task(self, task: Task) -> str:
        """Submits a task to the scheduler."""
        with self._lock:
            self._task_queue.put(task)
            self._task_status[task.task_id] = "pending"
            if task.dependencies:
                 self._completed_dependencies.setdefault(task.task_id, set())
            # print(f"Submitted task {task.name} ({task.task_id}) with priority {task.priority} and resources {task.requested_resources}")
        return task.task_id

    def get_task_status(self, task_id: str) -> Optional[str]:
        with self._lock:
            return self._task_status.get(task_id)

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Waits for a task to complete and returns its result."""
        start_time = time.time()
        while True:
            with self._lock:
                status = self._task_status.get(task_id)
                if status == "completed" or status == "failed":
                    return self._task_results.get(task_id)
                if status is None:
                    raise ValueError(f"Unknown task_id: {task_id}")
            
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id} to complete.")
            time.sleep(0.1)

    def shutdown(self, wait: bool = True):
        # print("Scheduler shutdown initiated.")
        self._stop_event.set()
        if wait:
            for worker in self._workers:
                worker.join()
        # print("Scheduler shutdown complete.")

# Example Usage (for direct testing)
if __name__ == "__main__":
    def sample_task_func(duration: float, task_name: str):
        print(f"Task '{task_name}' running for {duration}s on {threading.current_thread().name}...")
        time.sleep(duration)
        print(f"Task '{task_name}' finished.")
        return f"Result of {task_name}"

    def failing_task_func(task_name: str):
        print(f"Task '{task_name}' running on {threading.current_thread().name} and will fail...")
        time.sleep(0.5)
        raise ValueError(f"Error in {task_name}")

    scheduler = TaskScheduler(max_workers=2)

    # Task 1 (high priority, no deps, uses 1 CPU)
    task1_id = scheduler.submit_task(Task(
        priority=1, 
        name="HighPrio CPU Task",
        requested_resources={ResourceType.CPU: 1},
        func=sample_task_func, 
        args=(2, "HighPrio CPU Task")
    ))

    # Task 2 (medium priority, no deps, uses 1 CPU, 1 ANE - if ANE exists)
    # Check if ANE is available before requesting
    ane_available = scheduler._resource_manager.get_available_resources().get(ResourceType.ANE, 0) > 0
    res_task2 = {ResourceType.CPU: 1}
    if ane_available:
        res_task2[ResourceType.ANE] = 1
    
    task2_id = scheduler.submit_task(Task(
        priority=5, 
        name="MediumPrio CPU/ANE Task",
        requested_resources=res_task2,
        func=sample_task_func, 
        args=(3, "MediumPrio CPU/ANE Task")
    ))

    # Task 3 (low priority, depends on Task 1, uses 1 CPU)
    task3_id = scheduler.submit_task(Task(
        priority=10, 
        name="LowPrio Dependent Task",
        requested_resources={ResourceType.CPU: 1},
        func=sample_task_func, 
        args=(1, "LowPrio Dependent Task"),
        dependencies=[task1_id]
    ))
    
    # Task 4 (high priority, failing task)
    task4_id = scheduler.submit_task(Task(
        priority=2,
        name="Failing Task",
        requested_resources={ResourceType.CPU: 1},
        func=failing_task_func,
        args=("Failing Task",)
    ))
    
    # Task 5 (depends on failing task 4)
    task5_id = scheduler.submit_task(Task(
        priority=3,
        name="Task Depending on Failure",
        requested_resources={ResourceType.CPU: 1},
        func=sample_task_func,
        args=(1, "Task Depending on Failure"),
        dependencies=[task4_id]
    ))


    print(f"Task '{task1_id}' (HighPrio) status: {scheduler.get_task_status(task1_id)}")
    print(f"Task '{task2_id}' (MediumPrio) status: {scheduler.get_task_status(task2_id)}")
    print(f"Task '{task3_id}' (LowPrio) status: {scheduler.get_task_status(task3_id)}")
    print(f"Task '{task4_id}' (Failing) status: {scheduler.get_task_status(task4_id)}")
    print(f"Task '{task5_id}' (Dep Failing) status: {scheduler.get_task_status(task5_id)}")

    # Wait for results
    try:
        print(f"Result of Task 1: {scheduler.get_task_result(task1_id, timeout=10)}")
        print(f"Result of Task 2: {scheduler.get_task_result(task2_id, timeout=10)}")
        print(f"Result of Task 3: {scheduler.get_task_result(task3_id, timeout=10)}")
        task4_result = scheduler.get_task_result(task4_id, timeout=5)
        print(f"Result of Task 4: {task4_result}") 
        if isinstance(task4_result, Exception):
            print(f"Task 4 failed as expected: {type(task4_result).__name__}: {str(task4_result)}")
        print(f"Result of Task 5: {scheduler.get_task_result(task5_id, timeout=5)}")

    except TimeoutError as e:
        print(e)
    except ValueError as e:
        print(e)
    finally:
        print("Shutting down scheduler...")
        scheduler.shutdown()
        print("Scheduler finished.") 