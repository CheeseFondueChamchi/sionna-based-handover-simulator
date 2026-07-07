"""
Discrete Event Scheduler for NR Handover Simulation

This module provides a priority-queue based event scheduler that drives
the entire simulation. All time-based events (measurements, timers, 
position updates) are managed through this scheduler.
"""

import heapq
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, List, Dict
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the simulation"""
    # Measurement Events
    MEASUREMENT_PERIOD = auto()
    SYNC_INDICATION = auto()       # In-sync / Out-of-sync
    
    # Timer Events (3GPP TS 38.331)
    TTT_EXPIRE = auto()            # Time-to-Trigger for measurement events
    T300_EXPIRE = auto()           # RRC Connection Request
    T301_EXPIRE = auto()           # RRC Connection Re-establishment Request  
    T304_EXPIRE = auto()           # Handover execution
    T310_EXPIRE = auto()           # RLF detection
    T311_EXPIRE = auto()           # RRC re-establishment
    
    # Handover Events
    HO_PREPARATION = auto()
    HO_COMMAND = auto()
    HO_EXECUTION = auto()
    HO_COMPLETE = auto()
    HO_FAILURE = auto()
    
    # RACH Events
    RACH_PREAMBLE_TX = auto()
    RACH_RESPONSE = auto()
    RACH_SUCCESS = auto()
    RACH_FAILURE = auto()
    
    # RLF Events
    RLF_DETECTED = auto()
    RLF_RECOVERY = auto()
    
    # Channel/Position Events
    CHANNEL_UPDATE = auto()
    UE_POSITION_UPDATE = auto()
    
    # PHY Events
    BLER_UPDATE = auto()
    
    # Generic
    CUSTOM = auto()


@dataclass(order=True)
class Event:
    """
    Simulation event with priority ordering by time.
    
    Events are compared by time for heap ordering.
    Additional fields are excluded from comparison.
    """
    time: float
    priority: int = field(default=0, compare=True)  # Lower = higher priority
    event_type: EventType = field(compare=False, default=EventType.CUSTOM)
    callback: Callable = field(compare=False, default=lambda: None)
    args: tuple = field(compare=False, default=())
    kwargs: dict = field(compare=False, default_factory=dict)
    event_id: int = field(compare=False, default=0)
    cancelled: bool = field(compare=False, default=False)
    description: str = field(compare=False, default="")


class EventScheduler:
    """
    Priority-queue based discrete event scheduler.
    
    Features:
    - O(log n) event insertion and extraction
    - Event cancellation support
    - Event logging for analysis
    - Simulation time tracking
    
    Usage:
        scheduler = EventScheduler()
        scheduler.schedule(delay=1.0, event_type=EventType.MEASUREMENT_PERIOD, 
                          callback=measure_fn)
        scheduler.run_until(end_time=10.0)
    """
    
    def __init__(self):
        """Construct an empty discrete-event scheduler.

        Side effects:
            Allocates the event-queue heap, current_time clock, monotonic
            event counter (tiebreaker for events with equal timestamps),
            event log list, running flag, and per-type stats counter.
        """
        self.event_queue: List[Event] = []
        self.current_time: float = 0.0
        self._event_counter: int = 0
        self._event_log: List[Dict] = []
        self._running: bool = False
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'cancelled_events': 0,
            'events_by_type': {}
        }
    
    @property
    def event_log(self) -> List[Dict]:
        """Return event log for analysis"""
        return self._event_log
    
    def schedule(self, delay: float, event_type: EventType,
                 callback: Callable, *args, priority: int = 0,
                 description: str = "", **kwargs) -> Event:
        """
        Schedule an event to occur after 'delay' seconds.
        
        Args:
            delay: Time delay in seconds from current time
            event_type: Type of event (for logging and filtering)
            callback: Function to call when event fires
            *args: Positional arguments for callback
            priority: Event priority (lower = higher priority, for same-time events)
            description: Optional description for logging
            **kwargs: Keyword arguments for callback
            
        Returns:
            Event object (can be used for cancellation)
        """
        self._event_counter += 1
        
        event = Event(
            time=self.current_time + delay,
            priority=priority,
            event_type=event_type,
            callback=callback,
            args=args,
            kwargs=kwargs,
            event_id=self._event_counter,
            description=description
        )
        
        heapq.heappush(self.event_queue, event)
        
        logger.debug(f"Scheduled event {event.event_id}: {event_type.name} "
                    f"at t={event.time:.6f}s ({description})")
        
        return event
    
    def schedule_absolute(self, time: float, event_type: EventType,
                         callback: Callable, *args, priority: int = 0,
                         description: str = "", **kwargs) -> Event:
        """
        Schedule an event at an absolute time.
        
        Args:
            time: Absolute time for event
            (other args same as schedule())
            
        Returns:
            Event object
        """
        delay = time - self.current_time
        if delay < 0:
            logger.warning(f"Scheduling event in the past: t={time}, current={self.current_time}")
            delay = 0
        
        return self.schedule(delay, event_type, callback, *args,
                           priority=priority, description=description, **kwargs)
    
    def cancel(self, event: Event) -> bool:
        """
        Cancel a scheduled event.
        
        Note: Event is marked as cancelled but not removed from heap
        (lazy deletion for efficiency).
        
        Args:
            event: Event to cancel
            
        Returns:
            True if event was successfully cancelled
        """
        if event.cancelled:
            return False
        
        event.cancelled = True
        self.stats['cancelled_events'] += 1
        
        logger.debug(f"Cancelled event {event.event_id}: {event.event_type.name}")
        
        return True
    
    def run_until(self, end_time: float) -> int:
        """
        Run simulation until end_time.
        
        Args:
            end_time: Simulation end time in seconds
            
        Returns:
            Number of events processed
        """
        self._running = True
        events_processed = 0
        
        while self._running and self.event_queue:
            # Peek at next event
            next_event = self.event_queue[0]
            
            # Check if we've reached end time
            if next_event.time > end_time:
                break
            
            # Pop event
            event = heapq.heappop(self.event_queue)
            
            # Skip cancelled events
            if event.cancelled:
                continue
            
            # Update simulation time
            self.current_time = event.time
            
            # Log event
            self._log_event(event)
            
            # Execute callback
            try:
                event.callback(*event.args, **event.kwargs)
                events_processed += 1
                self.stats['total_events'] += 1
                
                # Track by type
                type_name = event.event_type.name
                self.stats['events_by_type'][type_name] = \
                    self.stats['events_by_type'].get(type_name, 0) + 1
                    
            except Exception as e:
                logger.error(f"Error executing event {event.event_id} "
                           f"({event.event_type.name}): {e}")
                raise
        
        self._running = False
        return events_processed
    
    def run_next(self) -> Optional[Event]:
        """
        Run single next event.
        
        Returns:
            The event that was processed, or None if queue is empty
        """
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            
            if event.cancelled:
                continue
            
            self.current_time = event.time
            self._log_event(event)
            
            event.callback(*event.args, **event.kwargs)
            self.stats['total_events'] += 1
            
            return event
        
        return None
    
    def stop(self):
        """Stop the simulation loop"""
        self._running = False
    
    def clear(self):
        """Clear all pending events"""
        self.event_queue.clear()
        self._event_counter = 0
    
    def _log_event(self, event: Event):
        """Log event for later analysis"""
        self._event_log.append({
            'time': event.time,
            'type': event.event_type.name,
            'id': event.event_id,
            'description': event.description
        })
    
    def get_pending_count(self) -> int:
        """Get number of pending (non-cancelled) events"""
        return sum(1 for e in self.event_queue if not e.cancelled)
    
    def peek_next_time(self) -> Optional[float]:
        """Peek at the time of the next event without removing it"""
        for event in self.event_queue:
            if not event.cancelled:
                return event.time
        return None
    
    def get_stats(self) -> Dict:
        """Get simulation statistics"""
        return {
            **self.stats,
            'current_time': self.current_time,
            'pending_events': self.get_pending_count()
        }


class Timer:
    """
    Named timer that can be started, stopped, and restarted.
    
    Wraps EventScheduler for convenient timer management.
    Supports 3GPP timer semantics (start, stop, expire).
    """
    
    def __init__(self, name: str, duration_ms: float, 
                 scheduler: EventScheduler,
                 on_expire: Optional[Callable] = None):
        """
        Initialize a timer.
        
        Args:
            name: Timer name (e.g., "T304", "T310")
            duration_ms: Timer duration in milliseconds
            scheduler: EventScheduler instance
            on_expire: Callback when timer expires
        """
        self.name = name
        self.duration_ms = duration_ms
        self.duration_s = duration_ms / 1000.0
        self.scheduler = scheduler
        self.on_expire = on_expire
        
        self._event: Optional[Event] = None
        self._running: bool = False
        self._start_time: Optional[float] = None
    
    @property
    def running(self) -> bool:
        """Check if timer is running"""
        return self._running
    
    @property
    def remaining_ms(self) -> Optional[float]:
        """Get remaining time in milliseconds"""
        if not self._running or self._event is None:
            return None
        remaining_s = self._event.time - self.scheduler.current_time
        return max(0, remaining_s * 1000)
    
    def start(self, on_expire: Optional[Callable] = None):
        """
        Start the timer.
        
        Args:
            on_expire: Optional override for expiry callback
        """
        if self._running:
            self.stop()
        
        callback = on_expire or self.on_expire
        if callback is None:
            raise ValueError(f"Timer {self.name}: No expiry callback defined")
        
        self._running = True
        self._start_time = self.scheduler.current_time
        
        # Determine event type from timer name
        event_type = self._get_event_type()
        
        self._event = self.scheduler.schedule(
            delay=self.duration_s,
            event_type=event_type,
            callback=self._expire,
            description=f"Timer {self.name} ({self.duration_ms}ms)"
        )
        
        logger.debug(f"Timer {self.name} started ({self.duration_ms}ms)")
    
    def stop(self) -> bool:
        """
        Stop the timer.
        
        Returns:
            True if timer was running and is now stopped
        """
        if not self._running:
            return False
        
        if self._event:
            self.scheduler.cancel(self._event)
        
        self._running = False
        self._event = None
        
        logger.debug(f"Timer {self.name} stopped")
        
        return True
    
    def restart(self, on_expire: Optional[Callable] = None):
        """Restart the timer (stop + start)"""
        self.stop()
        self.start(on_expire)
    
    def _expire(self):
        """Internal expiry handler"""
        self._running = False
        self._event = None
        
        logger.debug(f"Timer {self.name} expired")
        
        if self.on_expire:
            self.on_expire()
    
    def _get_event_type(self) -> EventType:
        """Map timer name to event type"""
        mapping = {
            'T300': EventType.T300_EXPIRE,
            'T301': EventType.T301_EXPIRE,
            'T304': EventType.T304_EXPIRE,
            'T310': EventType.T310_EXPIRE,
            'T311': EventType.T311_EXPIRE,
            'TTT': EventType.TTT_EXPIRE,
        }
        return mapping.get(self.name, EventType.CUSTOM)
