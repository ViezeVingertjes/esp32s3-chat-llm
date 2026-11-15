# ESP32-S3 chat LLM

A small conversational language model that runs on an ESP32-S3 microcontroller. The tokenizer,
the chat template, the conversation memory and the decoding all run on the board. You type on a
PC; the board answers over USB serial, word by word.

As far as we could find, a conversational LLM had never been run on an ESP32-S3 before. This
repository documents how we did it. It is a snapshot, not a maintained project.

## The model

- **Name:** Cp, shipped as `model/model-cp-chat-512.ellm` (fingerprint `fnv1a=0x07229bd6`).
- **Architecture:** 20 layers, width 576, feed-forward width 768, 8 query heads sharing one KV head,
  24,576-token vocabulary, 512-token context.
- **Weights:** 2 bits each. Each generated token uses about 54M weights. About 3.9 MB of them do not
  fit in PSRAM and are read from flash.
- **Base model (teacher):** [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) by Hugging Face (Apache-2.0). See Credits below.
- **Training data:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).
- **Training, three steps:** distillation from the teacher (2B tokens); chat-heavy 2-bit training
  (400M tokens); a chat polish on multi-turn conversations that the teacher regenerated. The cloud
  GPU time (Modal) came to about $38.
- **Decoding:** greedy, with a 4-gram ban inside a reply (no repeated 4-word phrase within the reply)
  and a 4-gram ban on copying from its own earlier replies. We picked these with a repetition sweep
  over eight test prompts: 4/4 scored best (mean share of repeated 3-word phrases 0.008, against 0.050
  for the earlier 8/3 setting).
- **Quality, honestly:** it handles small talk (greetings, thanks and goodbyes, names, simple facts).
  It often drifts into stock phrases and frequently refuses tasks. On IFEval it scores 20.8 against
  the teacher's 29.9, so it does not meet our instruction-following bar.

## Speed and memory (measured on the board)

| | |
|---|---|
| Generation at the start of a chat | ~2.6 tok/s (~2.0 words/s) |
| Generation at the 511th token of a full context | 2.13 tok/s (1.6 words/s) |
| First reply byte | 3-6 s for short prompts |
| Boot until the prompt | ~6.6 s |
| Free memory after boot | 451 KB PSRAM, 186 KB internal RAM |

One token is about 0.75 words. The speed figures include the status screen, which costs under 1%.

## How we got it running

1. **Measure the hardware first.** We benchmarked the 2-bit lookup-table matrix-vector kernel on
   the board before choosing a model size.
2. **Shrink the model.** SmolLM2-135M-Instruct was the teacher. We cut its vocabulary from 49,152
   to 24,576 tokens, choosing the kept tokens by a bits-per-character proxy and checking the result
   with IFEval. The student has 20 layers and width 576.
3. **Quantize to 2 bits** and train with quantization-aware training under distillation.
4. **One C runtime for the PC and the board.** The same engine (`runtime/`) runs on both. Host and
   board produce bit-identical logits and byte-identical chat replies.
5. **Make it fast on the S3.** The weights stay in a pack mapped from flash. Activations and the KV
   cache live in PSRAM. The matrix-vector and attention inner loops use the ESP32-S3's Xtensa PIE
   SIMD instructions in assembly. Both cores share every matrix-vector product and the attention.
   The LM head is skipped while the prompt is read.
6. **Put the tokenizer on the board.** The model file carries the tokenizer, so the board needs no
   host-side preprocessing.
7. **Chat over USB serial.** A small Python client talks to the board. A 300 s soak test passed.

## Run it

You need an ESP32-S3 board with 16 MB flash and octal PSRAM. The firmware was built for the
Waveshare ESP32-S3-Touch-AMOLED-2.16, and its status-screen code is specific to that board's
display, so other boards are untested. You also need ESP-IDF v6.1 and Python 3 with `pyserial`.

```
. ~/esp/esp-idf-v6.1/export.sh

# 1. Flash the model (once, about 45 s)
esptool.py --chip esp32s3 -p /dev/ttyACM0 -b 921600 write_flash 0x190000 model/model-cp-chat-512.ellm

# 2. Flash the firmware
cd firmware/chat
idf.py set-target esp32s3
idf.py -p /dev/ttyACM0 build flash
cd ../..

# 3. Chat
pip install pyserial
python3 tools/chat.py /dev/ttyACM0
```

The board prints `ELLM chat ready: pack fnv1a=0x07229bd6 ...` about 7 s after power-on, then the
prompt. Type a line and press Enter. `/reset` starts a new conversation. Ctrl-C stops a reply, and
Ctrl-D exits. After each reply the board prints a stats line such as `[69 tok, 4.27 tok/s]`.

The conversation memory is 512 tokens. When it fills, the oldest turns are dropped. A message can be
up to 1,024 bytes.

### Example

A short session on the board (`tools/chat.py`; lines starting with `>` are typed by the user):

```
> Hello, how are you?
Hello! How can I help you today?
[9 tok, 2.63 tok/s]
> What is the capital of Germany?
The capital of Germany is Berlin.
[7 tok, 2.59 tok/s]
> What is the largest planet?
The largest planet is Earth, which is about 23,000 miles (56,009 kilometers) away from Earth.
[30 tok, 2.55 tok/s]
> How is your day going?
I'm doing well, thanks for asking. I'm doing great, and I'm happy to help you with any questions or concerns you have. I'm here to assist you in any way I can, whether it's a simple task, a simple problem, or just a quick update on your current situation.
[62 tok, 2.59 tok/s]
> Tell me something nice.
I'm glad you're enjoying my day. I'm glad you found my response helpful. I'm happy with any questions you have, and I'll do my best to help you find the perfect solution.
[41 tok, 2.51 tok/s]
> Thanks, that is all for now.
I'm glad I could help you find a solution. If you have any more questions or need further assistance, feel free to ask.
[27 tok, 2.44 tok/s]
```

The model is built for basic conversation. Greetings, thanks and goodbyes work, and it answers
in complete sentences. It is wordy and sometimes misses the point, as in the replies above. Its
factual answers are unreliable: the capital is right, the planet answer is wrong.

## Related work

Two other projects run TinyStories models on an ESP32-S3. Both generate stories, not chat.

| | This repo (Cp) | [JARACH-209/esp32-30.7M](https://github.com/JARACH-209/esp32-30.7M) | [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) |
|---|---|---|---|
| Task | chat | TinyStories | TinyStories |
| Stored parameters | 54.0M | 30.72M | 28.9M (25M in a flash lookup table) |
| Multiplied per token | 54.0M | 30.72M | ~3.9M (13%) |
| Weight precision | 2-bit | 4-bit (group 128) | 4-bit |
| Weight bytes read per token | ~13.5 MB (before scales) | 15.4 MB | ~4.5 MB |
| Decode speed | 2.6 tok/s at chat start, 2.13 at token 511 (prompt excluded) | 0.95 tok/s end to end | 9.88 tok/s end to end |

The other two figures are from their READMEs. Ours are from this repo's board runs. The rows are not
measured the same way, so read them as rough context, not a ranking:

- **Parameter count depends on bit width.** A 2-bit weight carries less than a 4-bit one, so a
  54M-parameter 2-bit model is not "bigger" than a 30.7M 4-bit model in any sense that matters for
  quality. The byte row is the fairer size measure: by bytes read per token, this model is slightly
  smaller than JARACH's, and much larger than slvDev's, which reads few bytes because of its lookup table.
- **Speed is not measured the same way.** Our figure excludes the prompt (the stats line counts only
  the reply). JARACH and slvDev report end-to-end rates. We have not measured our end-to-end rate, so
  the true figure is lower than 2.6 tok/s.
- **Different hardware.** JARACH uses an ESP32-S3 N16R8; we use the Waveshare AMOLED board.
- **Quality is not compared.** The tasks differ (chat against stories), and we have not run their
  evaluations.

## Layout

- `firmware/chat/`: the chat firmware (ESP-IDF project).
- `firmware/components/`: `ellm` (dual-core matvec and attention), `ellm_stage` (pack loading and
  PSRAM staging), `status_display` (the AMOLED screen; `DejaVu-LICENSE.txt` covers its font).
- `runtime/`: the portable C inference engine, tokenizer and chat engine, shared by the board and
  the PC.
- `tools/chat.py`: the serial chat client.
- `model/model-cp-chat-512.ellm`: the model file.

## Credits

- **Base model:** [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
  by Hugging Face (Apache-2.0). Our model is distilled from it, with a trimmed vocabulary and 2-bit
  weights. The chat-polish conversations were regenerated by the same teacher.
- **Training data:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (ODC-By 1.0) and [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (Apache-2.0).
- **Display:** pin mapping, panel initialisation and orientation follow the Waveshare ESP32-S3-Touch-AMOLED-2.16
  BSP (v2.0.1), the CO5300 datasheet and the esp_lcd_co5300 driver, as cited in
  `firmware/components/status_display/status_display.c`. The font is DejaVu (see its license file).
- **Tooling:** [ESP-IDF](https://github.com/espressif/esp-idf) (Espressif), [pyserial](https://github.com/pyserial/pyserial),
  and esptool.
