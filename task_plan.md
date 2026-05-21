# Task Plan: Vehicle Color Detection Integration

## Goal
Integrate rule-based vehicle color detection into the DeepStream detection pipeline so that each published detection payload includes a "color" field.

## Current Phase
Phase 5 — Delivery

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand existing codebase structure (config, vehicle_color_rule, detect_pipeline, common)
- [x] Identify all files to modify and changes needed
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Plan data flow: probe stores entries → appsink crops & detects color → enriched batch published
- [x] Plan thread safety: _cam_color_lock for queue/enriched dicts
- [x] Plan edge cases: non-vehicle objects, camera lag/drop, zero-size crops
- **Status:** complete

### Phase 3: Implementation
- [x] Add VEHICLE_CLASS_IDS to config.py
- [x] Update detect_pipeline.py imports
- [x] Add __init__ fields
- [x] Modify _tracked_src_pad_buffer_probe (cycle, queue, remove publish)
- [x] Add _on_camera_sample override
- [x] Add _process_camera_colors method
- [x] Add _try_publish_enriched_batch method
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Verify syntax with Python import check
- [x] Verify no regressions in existing logic
- **Status:** complete

### Phase 5: Delivery
- [x] Review all changes
- [x] Report to user
- **Status:** complete

## Key Questions
1. Thread safety: probe and appsink callbacks run on different GStreamer threads → use _cam_color_lock ✓
2. Coordinate system: probe bbox and appsink frame both use MUX_WIDTH x MUX_HEIGHT → match directly ✓
3. Source ID matching: frame_meta.source_id == cam_idx == demux src index → matches ✓

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use simple dict overwrite for _cam_color_queue | Probe fires per batch; overwriting is acceptable for real-time |
| Include 0.5s timeout in _try_publish_enriched_batch | Prevents stall if one camera drops |
| Keep raw_data_batch local build in probe | References stored in _cam_color_queue; enriched entries published later |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
|       |         |            |

## Notes
- vehicle_color_rule.py already handles None/empty inputs (returns "Unknown")
- MJPEG streaming per camera is preserved alongside color processing
