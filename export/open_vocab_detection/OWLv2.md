# OWLv2 Export Guide

## Supported Contract

The `neuriplo-tasks` OWLv2 path expects a Hugging Face style ONNX model with:

- Inputs:
  - `pixel_values` of shape `[batch, 3, height, width]`
  - `input_ids` of shape `[num_queries, sequence_length]`
  - `attention_mask` of shape `[num_queries, sequence_length]`
- Outputs:
  - `logits` of shape `[batch, num_patches, num_queries]`
  - `objectness_logits` of shape `[batch, num_patches]` (optional but recommended)
  - `pred_boxes` of shape `[batch, num_patches, 4]`, normalized `(cx, cy, w, h)`

`neuriplo-tasks` resolves detections against runtime text prompts and maps the winning prompt index back to the corresponding label.

## Export

```bash
python export/open_vocab_detection/owlv2/export_owlv2_to_onnx.py \
  --model google/owlv2-base-patch16-ensemble \
  --output owlv2.onnx \
  --image-height 960 \
  --image-width 960 \
  --max-queries 16 \
  --sequence-length 16 \
  --test
```

## Notes

- The host application should provide tokenizer assets to `neuriplo-tasks` at runtime.
  - Preferred: pass preloaded `vocab.json` and `merges.txt` contents through `TaskConfig`
  - Fallback: pass file paths if your app manages assets on disk
- `pred_boxes` are normalized relative to the original image size and are converted to pixel coordinates in `neuriplo-tasks`.
- If your exported model uses different output names, keep them semantically equivalent to:
  - `logits`
  - `objectness_logits`
  - `pred_boxes`
