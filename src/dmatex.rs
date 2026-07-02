use timeline_syncobj::timeline_syncobj::TimelineSyncObj;

struct DmaImage {
    image: vulkano::image::Image,
    timeline: TimelineSyncObj,
}
