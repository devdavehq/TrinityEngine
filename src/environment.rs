// src/environment.rs — Environment module declarations.
// The environment system controls the sky, sun, weather, clouds, lightning,
// and all atmospheric effects. It follows the Bible's "data-driven, compositional"
// philosophy: each subsystem is a plain data struct that the renderer reads
// to produce the final image.

pub mod time_of_day;
pub mod sky;
pub mod weather;
pub mod clouds;
pub mod lightning;
pub mod weather_zone;
pub mod wind_zone;
pub mod splash;
pub mod flood;
