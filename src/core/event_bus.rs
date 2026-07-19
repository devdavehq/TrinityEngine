// src/core/event_bus.rs
// ──────────────────────────────────────────────────────────────────────────────
// The EventBus is the nervous system of the engine.
//
// WHY IT EXISTS:
//   Without an event bus, every system talks to every other system through
//   direct function calls and shared references. That creates spaghetti:
//   physics needs to know about scripting, scripting needs to know about
//   the renderer, the editor needs to know about physics, and so on.
//
//   The event bus lets systems communicate WITHOUT knowing about each other.
//   Physics emits "two entities collided." The audio system listens for that
//   and plays a sound. The scripting system listens and calls Lua callbacks.
//   The editor listens and highlights the collision in the viewport.
//   None of those systems know the others exist.
//
// HOW AAA ENGINES USE IT:
//   Unreal: delegates + event dispatchers
//   Unity:  C# events / UnityEvent
//   Frostbite: global event bus with priority queues
//   Our version: type-safe, zero-allocation dispatch, deferred + immediate
//
// DESIGN RULES:
//   1. Events are plain data structs. No logic in events.
//   2. Handlers are closures registered per event type.
//   3. emit() queues. flush() dispatches the queue.
//   4. dispatch() fires immediately (bypass queue).
//   5. Events must be Send + Sync for future multithreaded dispatch.
// ──────────────────────────────────────────────────────────────────────────────

use std::any::{Any, TypeId};
use std::collections::{HashMap, VecDeque};

// ── Handler storage ───────────────────────────────────────────────────────────
// A handler is any Fn(&E) that is Send + Sync.
// We type-erase it because we store handlers for many different event types
// in the same HashMap. At dispatch time we downcast back.

type HandlerFn = Box<dyn Fn(&dyn Any) + Send + Sync>;

struct HandlerList {
    handlers: Vec<HandlerFn>,
}

impl HandlerList {
    fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    fn push(&mut self, h: HandlerFn) {
        self.handlers.push(h);
    }

    fn dispatch(&self, event: &dyn Any) {
        for h in &self.handlers {
            h(event);
        }
    }

    fn len(&self) -> usize {
        self.handlers.len()
    }
}

// ── EventBus ──────────────────────────────────────────────────────────────────
// Central event dispatcher. One per engine. Owned by GameApp.
//
// Usage:
//   // Register a listener
//   event_bus.subscribe::<CollisionEvent>(|e| {
//       tracing::info!("{} hit {}", e.entity_a, e.entity_b);
//   });
//
//   // Queue an event (dispatched next flush)
//   event_bus.emit(CollisionEvent { entity_a, entity_b, .. });
//
//   // Process all queued events (called once per frame)
//   event_bus.flush();
//
//   // Immediate dispatch (bypasses queue, for urgent events)
//   event_bus.dispatch(&ShutdownEvent);

pub struct EventBus {
    // Keyed by TypeId of the event struct.
    // Each TypeId maps to a list of handlers for that event type.
    handlers: HashMap<TypeId, HandlerList>,

    // Deferred event queue. emit() pushes here, flush() drains.
    queue: VecDeque<Box<dyn Any + Send + Sync>>,

    // Stats for the profiler
    pub events_dispatched_this_frame: u32,
    pub events_queued_this_frame: u32,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            queue: VecDeque::new(),
            events_dispatched_this_frame: 0,
            events_queued_this_frame: 0,
        }
    }

    // ── subscribe ─────────────────────────────────────────────────────────
    // Register a handler for event type E.
    // Multiple handlers can listen for the same event.
    // Handlers fire in registration order.
    pub fn subscribe<E: 'static + Send + Sync>(
        &mut self,
        handler: impl Fn(&E) + Send + Sync + 'static,
    ) {
        let type_id = TypeId::of::<E>();

        // Type-erase: wrap the typed handler in a closure that downcasts.
        let wrapped: HandlerFn = Box::new(move |any_event: &dyn Any| {
            if let Some(typed) = any_event.downcast_ref::<E>() {
                handler(typed);
            }
        });

        self.handlers
            .entry(type_id)
            .or_insert_with(HandlerList::new)
            .push(wrapped);
    }

    // ── emit (deferred) ───────────────────────────────────────────────────
    // Queue an event for processing on the next flush().
    // This is the safe, common path. Most events should go through here.
    pub fn emit<E: 'static + Send + Sync>(&mut self, event: E) {
        self.queue.push_back(Box::new(event));
        self.events_queued_this_frame += 1;
    }

    // ── dispatch (immediate) ──────────────────────────────────────────────
    // Fire all handlers for this event RIGHT NOW. Bypasses the queue.
    // Use for urgent events like Shutdown or critical errors.
    pub fn dispatch<E: 'static + Send + Sync>(&self, event: &E) {
        let type_id = TypeId::of::<E>();
        if let Some(list) = self.handlers.get(&type_id) {
            list.dispatch(event);
        }
    }

    // ── flush ─────────────────────────────────────────────────────────────
    // Process all deferred events. Call once per frame at a well-defined point.
    // After flush(), the queue is empty.
    pub fn flush(&mut self) {
        self.events_dispatched_this_frame = 0;

        // Take ownership of the queue so handlers can emit new events
        // during dispatch (those go to a fresh queue we swap in next frame).
        let mut events: VecDeque<Box<dyn Any + Send + Sync>> = VecDeque::new();
        std::mem::swap(&mut self.queue, &mut events);

        for event in events.drain(..) {
            // Explicit coercion: &(dyn Any + Send + Sync) → &dyn Any.
            // This is needed because the handler storage is keyed by &dyn Any.
            let event_ref: &dyn Any = &*event;
            let type_id = event_ref.type_id();
            if let Some(list) = self.handlers.get(&type_id) {
                list.dispatch(event_ref);
            }
            self.events_dispatched_this_frame += 1;
        }
    }

    // ── clear ─────────────────────────────────────────────────────────────
    // Remove all handlers and queued events. Used on project switch.
    pub fn clear(&mut self) {
        self.handlers.clear();
        self.queue.clear();
    }

    // ── handler_count ─────────────────────────────────────────────────────
    // How many handlers are registered for event type E?
    pub fn handler_count<E: 'static + Send + Sync>(&self) -> usize {
        self.handlers
            .get(&TypeId::of::<E>())
            .map_or(0, |l| l.len())
    }

    // ── reset_frame_stats ─────────────────────────────────────────────────
    // Called at the start of each frame by the profiler.
    pub fn reset_frame_stats(&mut self) {
        self.events_dispatched_this_frame = 0;
        self.events_queued_this_frame = 0;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestEvent(u32);

    #[test]
    fn subscribe_and_emit() {
        let mut bus = EventBus::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&counter);

        bus.subscribe::<TestEvent>(move |e| {
            counter_clone.fetch_add(e.0, Ordering::Relaxed);
        });

        bus.emit(TestEvent(5));
        bus.emit(TestEvent(3));
        bus.flush();

        assert_eq!(counter.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn dispatch_immediate() {
        let mut bus = EventBus::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&counter);

        bus.subscribe::<TestEvent>(move |e| {
            counter_clone.fetch_add(e.0, Ordering::Relaxed);
        });

        bus.dispatch(&TestEvent(42));
        assert_eq!(counter.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn multiple_handlers() {
        let mut bus = EventBus::new();
        let a = Arc::new(AtomicU32::new(0));
        let b = Arc::new(AtomicU32::new(0));
        let a_clone = Arc::clone(&a);
        let b_clone = Arc::clone(&b);

        bus.subscribe::<TestEvent>(move |e| {
            a_clone.fetch_add(e.0, Ordering::Relaxed);
        });
        bus.subscribe::<TestEvent>(move |e| {
            b_clone.fetch_add(e.0 * 2, Ordering::Relaxed);
        });

        bus.emit(TestEvent(5));
        bus.flush();

        assert_eq!(a.load(Ordering::Relaxed), 5);
        assert_eq!(b.load(Ordering::Relaxed), 10);
    }
}
