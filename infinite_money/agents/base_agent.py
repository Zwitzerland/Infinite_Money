"""
Base agent class for Infinite_Money trading system.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid

from ..utils.logger import get_logger, log_agent_start, log_agent_stop
from ..utils.config import Config


@dataclass
class AgentState:
    """Agent state information."""
    agent_id: str
    name: str
    status: str = "idle"  # idle, running, stopped, error
    start_time: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class Task:
    """Task definition for agents."""
    task_id: str
    task_type: str
    priority: int = 1  # 1-10, higher is more important
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on


class BaseAgent(ABC):
    """
    Base class for all agents in the Infinite_Money system.
    
    Provides common functionality for:
    - Task management and execution
    - State tracking and monitoring
    - Error handling and recovery
    - Performance metrics collection
    - Inter-agent communication
    """
    
    def __init__(self, name: str, config: Config, agent_config: Dict[str, Any] = None):
        """Initialize base agent."""
        self.name = name
        self.config = config
        self.agent_config = agent_config or {}
        
        # Generate unique agent ID
        self.agent_id = f"{name}_{uuid.uuid4().hex[:8]}"
        
        # Initialize state
        self.state = AgentState(
            agent_id=self.agent_id,
            name=self.name
        )
        
        # Setup logger
        self.logger = get_logger(self.name)
        
        # Task management
        self.task_queue: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.start_time = None
        
        # Communication channels
        self.message_queue: List[Dict[str, Any]] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize agent-specific components
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize agent-specific components. Override in subclasses."""
        pass
    
    async def start(self) -> bool:
        """Start the agent."""
        try:
            log_agent_start(self.name, self.agent_config)
            
            self.state.status = "running"
            self.state.start_time = datetime.utcnow()
            self.start_time = time.time()
            
            # Start the main agent loop
            await self._run_agent_loop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start agent {self.name}: {str(e)}")
            self.state.status = "error"
            self.state.last_error = str(e)
            return False
    
    async def stop(self) -> bool:
        """Stop the agent gracefully."""
        try:
            self.logger.info(f"Stopping agent {self.name}")
            
            # Complete current tasks
            await self._complete_running_tasks()
            
            # Update state
            runtime = time.time() - self.start_time if self.start_time else 0
            self.state.status = "stopped"
            self.state.last_activity = datetime.utcnow()
            
            log_agent_stop(self.name, runtime)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping agent {self.name}: {str(e)}")
            return False
    
    async def _run_agent_loop(self):
        """Main agent execution loop."""
        while self.state.status == "running":
            try:
                # Process incoming messages
                await self._process_messages()
                
                # Execute pending tasks
                await self._execute_tasks()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check for stop condition
                if self._should_stop():
                    break
                
                # Sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in agent loop: {str(e)}")
                self.state.error_count += 1
                self.state.last_error = str(e)
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_messages(self):
        """Process incoming messages."""
        while self.message_queue:
            message = self.message_queue.pop(0)
            await self._handle_message(message)
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message. Override in subclasses."""
        message_type = message.get("type")
        if message_type in self.event_handlers:
            for handler in self.event_handlers[message_type]:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Error in message handler: {str(e)}")
    
    async def _execute_tasks(self):
        """Execute pending tasks."""
        # Sort tasks by priority and deadline
        self.task_queue.sort(key=lambda t: (t.priority, t.deadline or datetime.max))
        
        # Execute tasks that are ready
        ready_tasks = [t for t in self.task_queue if self._is_task_ready(t)]
        
        for task in ready_tasks[:self.config.system.max_concurrent_tasks]:
            if task.task_id not in self.running_tasks:
                await self._execute_task(task)
    
    def _is_task_ready(self, task: Task) -> bool:
        """Check if task is ready to execute."""
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        # Check deadline
        if task.deadline and datetime.utcnow() > task.deadline:
            return False
        
        return True
    
    async def _execute_task(self, task: Task):
        """Execute a single task."""
        try:
            self.logger.info(f"Executing task {task.task_id}: {task.task_type}")
            
            # Move to running tasks
            self.task_queue.remove(task)
            self.running_tasks[task.task_id] = task
            
            # Execute task
            result = await self.execute_task(task)
            
            # Handle result
            if result.get("status") == "success":
                self.completed_tasks[task.task_id] = task
                self.state.tasks_completed += 1
                self.logger.info(f"Task {task.task_id} completed successfully")
            else:
                await self._handle_task_failure(task, result)
            
        except Exception as e:
            await self._handle_task_failure(task, {"error": str(e)})
    
    async def _handle_task_failure(self, task: Task, result: Dict[str, Any]):
        """Handle task failure."""
        self.logger.error(f"Task {task.task_id} failed: {result}")
        
        task.retry_count += 1
        if task.retry_count < task.max_retries:
            # Retry task
            self.task_queue.append(task)
            self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
        else:
            # Mark as failed
            self.failed_tasks[task.task_id] = task
            self.state.tasks_failed += 1
            self.state.error_count += 1
        
        # Remove from running tasks
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
    
    async def _complete_running_tasks(self):
        """Complete currently running tasks."""
        for task in list(self.running_tasks.values()):
            try:
                # Give tasks a chance to complete
                await asyncio.wait_for(self._execute_task(task), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning(f"Task {task.task_id} timed out during shutdown")
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        current_time = datetime.utcnow()
        
        metrics = {
            "timestamp": current_time.isoformat(),
            "agent_id": self.agent_id,
            "status": self.state.status,
            "tasks_completed": self.state.tasks_completed,
            "tasks_failed": self.state.tasks_failed,
            "error_count": self.state.error_count,
            "queue_size": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
        }
        
        # Add agent-specific metrics
        agent_metrics = await self._get_agent_metrics()
        metrics.update(agent_metrics)
        
        self.performance_history.append(metrics)
        self.state.performance_metrics = metrics
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent-specific metrics. Override in subclasses."""
        return {}
    
    def _should_stop(self) -> bool:
        """Check if agent should stop. Override in subclasses."""
        return False
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task. Must be implemented by subclasses."""
        pass
    
    # Task management methods
    def add_task(self, task_type: str, data: Dict[str, Any] = None, 
                 priority: int = 1, deadline: datetime = None,
                 dependencies: List[str] = None) -> str:
        """Add a new task to the queue."""
        task_id = f"{self.name}_{uuid.uuid4().hex[:8]}"
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data or {},
            deadline=deadline,
            dependencies=dependencies or []
        )
        
        self.task_queue.append(task)
        self.logger.info(f"Added task {task_id}: {task_type}")
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a specific task."""
        if task_id in self.running_tasks:
            return "running"
        elif task_id in self.completed_tasks:
            return "completed"
        elif task_id in self.failed_tasks:
            return "failed"
        else:
            # Check if in queue
            for task in self.task_queue:
                if task.task_id == task_id:
                    return "queued"
        return None
    
    # Communication methods
    def send_message(self, target_agent: str, message_type: str, data: Dict[str, Any] = None):
        """Send message to another agent."""
        message = {
            "from": self.name,
            "to": target_agent,
            "type": message_type,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # In a real implementation, this would go through a message broker
        self.logger.info(f"Sending message to {target_agent}: {message_type}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    # State and monitoring methods
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-100:]
        
        return {
            "total_tasks": self.state.tasks_completed + self.state.tasks_failed,
            "success_rate": self.state.tasks_completed / max(1, self.state.tasks_completed + self.state.tasks_failed),
            "error_rate": self.state.error_count / max(1, len(self.performance_history)),
            "avg_queue_size": sum(m["queue_size"] for m in recent_metrics) / len(recent_metrics),
            "uptime_seconds": (datetime.utcnow() - self.state.start_time).total_seconds() if self.state.start_time else 0
        }
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        return (
            self.state.status in ["running", "idle"] and
            self.state.error_count < 10 and
            len(self.performance_history) > 0
        )