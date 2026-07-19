use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum BlackboardValue {
    Float(f32),
    Vec3([f32; 3]),
    Entity(u64),
    Bool(bool),
    String(String),
    Path(Vec<[f32; 3]>),
}

#[derive(Debug, Clone, Default)]
pub struct Blackboard {
    values: HashMap<String, BlackboardValue>,
}

impl Blackboard {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: &str, value: BlackboardValue) {
        self.values.insert(key.to_string(), value);
    }

    pub fn has(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    pub fn remove(&mut self, key: &str) {
        self.values.remove(key);
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    pub fn get_float(&self, key: &str) -> Option<f32> {
        match self.values.get(key) {
            Some(BlackboardValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_vec3(&self, key: &str) -> Option<[f32; 3]> {
        match self.values.get(key) {
            Some(BlackboardValue::Vec3(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_entity(&self, key: &str) -> Option<u64> {
        match self.values.get(key) {
            Some(BlackboardValue::Entity(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.values.get(key) {
            Some(BlackboardValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.values.get(key) {
            Some(BlackboardValue::String(v)) => Some(v.as_str()),
            _ => None,
        }
    }

    pub fn get_path(&self, key: &str) -> Option<&Vec<[f32; 3]>> {
        match self.values.get(key) {
            Some(BlackboardValue::Path(v)) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blackboard_get_set_float() {
        let mut bb = Blackboard::new();
        assert!(!bb.has("hp"));
        bb.set("hp", BlackboardValue::Float(100.0));
        assert!(bb.has("hp"));
        assert_eq!(bb.get_float("hp"), Some(100.0));
    }

    #[test]
    fn blackboard_get_set_vec3() {
        let mut bb = Blackboard::new();
        bb.set("target", BlackboardValue::Vec3([1.0, 2.0, 3.0]));
        assert_eq!(bb.get_vec3("target"), Some([1.0, 2.0, 3.0]));
    }

    #[test]
    fn blackboard_remove_and_clear() {
        let mut bb = Blackboard::new();
        bb.set("a", BlackboardValue::Bool(true));
        bb.set("b", BlackboardValue::Float(5.0));
        bb.remove("a");
        assert!(!bb.has("a"));
        assert!(bb.has("b"));
        bb.clear();
        assert!(!bb.has("b"));
    }

    #[test]
    fn blackboard_type_mismatch_returns_none() {
        let mut bb = Blackboard::new();
        bb.set("val", BlackboardValue::Float(1.0));
        assert_eq!(bb.get_bool("val"), None);
        assert_eq!(bb.get_vec3("val"), None);
    }

    #[test]
    fn blackboard_string_and_path() {
        let mut bb = Blackboard::new();
        bb.set("name", BlackboardValue::String("guard".to_string()));
        assert_eq!(bb.get_string("name"), Some("guard"));

        let waypoints = vec![[0.0, 0.0, 0.0], [5.0, 0.0, 5.0]];
        bb.set("patrol", BlackboardValue::Path(waypoints.clone()));
        assert_eq!(bb.get_path("patrol"), Some(&waypoints));
    }

    #[test]
    fn blackboard_entity() {
        let mut bb = Blackboard::new();
        bb.set("owner", BlackboardValue::Entity(42));
        assert_eq!(bb.get_entity("owner"), Some(42));
    }
}
