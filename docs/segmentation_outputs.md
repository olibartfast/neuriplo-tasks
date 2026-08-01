# Instance-segmentation output representations

`InstanceSegmentationTask` can return either masks or polygons. Masks remain the
default for backward compatibility.

## Selecting the representation

Set `TaskConfig::segmentation_output` before creating the task:

```cpp
#include <neuriplo/tasks/core/task_factory.hpp>

neuriplo_tasks::TaskConfig config;
config.segmentation_output = neuriplo_tasks::SegmentationOutput::Polygon;

auto task = neuriplo_tasks::TaskFactory::createTaskInstance(
    "yolo26seg", model_info, config);
```

The option applies to all instance-segmentation task families routed through
`TaskFactory`, including YOLO segmentation, RF-DETR segmentation, and
EdgeCrafter segmentation.

## Result contract

`InstanceSegmentation` always contains the detection class, confidence, and
bounding box. Its representation-specific fields are:

| Mode | Populated fields | Empty fields |
|---|---|---|
| `SegmentationOutput::Mask` | `mask` and model-specific mask metadata | `polygons` |
| `SegmentationOutput::Polygon` | `polygons` | `mask`, `mask_data`, `mask_height`, `mask_width` |

Polygon mode enforces one representation per result: after conversion, mask
storage is released and its dimensions are reset to zero.

Each `SegmentationPolygon` contains:

- `exterior`: the exterior boundary in full-image pixel coordinates;
- `holes`: zero or more interior boundary rings;
- three or more points per ring, without repeating the first point at the end.

Coordinates use `vision::Point2f` so the public API remains framework-neutral.
Mask conversion produces integer-valued pixel-boundary coordinates and applies a
convex hull to every exterior and hole ring. In image coordinates, where Y increases downward, exterior rings
have positive signed area and hole rings have negative signed area. A single
instance may contain multiple polygons when its mask has disconnected regions.

## Direct conversion

Consumers using postprocessors directly can convert a single-channel `UINT8`
`ImageMatrix` without creating a task:

```cpp
#include <neuriplo/tasks/instance_segmentation/polygon_conversion.hpp>

std::vector<neuriplo_tasks::SegmentationPolygon> polygons =
    neuriplo_tasks::maskToPolygons(mask);
```

The converter traces foreground pixel boundaries, computes a convex hull for each
ring, preserves disconnected regions and holes, and rejects non-`UINT8` or
multi-channel inputs. Convexification intentionally removes concave boundary
detail to provide a compact polygon contract.

## Engine and transport boundary

`neuriplo-tasks` defines the in-process result contract. Inference transports
with variable-length tensor outputs should encode polygons with flattened point
arrays plus offsets, then decode them into `SegmentationPolygon`. Tensor names,
padding limits, and transport serialization remain consumer responsibilities.
