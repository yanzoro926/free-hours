#!/usr/bin/env python3
"""
Synthetic Seashore / 合成潮汐

A Python reimagining of tompng's IOCCC 2025 winning entry
"Synthetic seashore." Generates a stereo WAV file of procedurally
layered ocean waves with ambient tones — each run produces a
unique seashore.

Algorithm:
  1. Generate individual "bubble" impulses with random parameters
  2. Layer thousands of bubbles into wave clusters
  3. Apply bandpass filtering for ocean-like frequency response
  4. Mix multiple wave layers at different scales
  5. Add ambient musical tones that blend into the water sound

Inspired by: https://www.ioccc.org/2025/tompng/

Technical note: This uses only Python stdlib — no numpy, no scipy.
Pure math, pure sound.
"""

import struct
import wave
import math
import random
import sys
import os


class SeashoreGenerator:
    """Procedural ocean wave and ambient tone generator."""

    SAMPLE_RATE = 44100
    BITS_PER_SAMPLE = 16
    NUM_CHANNELS = 2  # stereo

    def __init__(self, seed=None):
        if seed is None:
            seed = int(os.urandom(4).hex(), 16) % 2**31
        random.seed(seed)
        self.seed = seed

    def _generate_impulse(self, duration_samples, center_freq, bandwidth, decay_rate):
        """
        Generate a single wave impulse — a frequency-modulated
        decaying sine wave that mimics a bubble burst.
        """
        samples = []
        for i in range(duration_samples):
            t = i / self.SAMPLE_RATE

            # Decaying envelope
            envelope = math.exp(-decay_rate * t)

            # Frequency sweep (bubble pitch drops as it decays)
            freq = center_freq * (1.0 - 0.3 * (1.0 - math.exp(-decay_rate * 2 * t)))

            # Add slight frequency modulation for organic feel
            freq *= 1.0 + 0.02 * math.sin(2 * math.pi * 12 * t)

            phase = 2 * math.pi * freq * t
            # Bandpass-like shaping
            value = envelope * math.sin(phase)

            # Add harmonic for richness
            value += 0.15 * envelope * math.sin(phase * 2.3)

            samples.append(value)

        return samples

    def _layer_waves(self, num_impulses, duration_sec, base_freq, freq_spread,
                     decay_range, density):
        """
        Create a layer of wave sound by combining many impulses
        with random timing, frequency, and decay.
        """
        total_samples = int(duration_sec * self.SAMPLE_RATE)
        output = [0.0] * total_samples

        for _ in range(num_impulses):
            # Random impulse parameters
            center_freq = base_freq + random.uniform(-freq_spread, freq_spread)
            center_freq = max(50, min(8000, center_freq))

            # Duration: short for high freq, longer for low
            impulse_dur = random.uniform(0.02, 0.3) * (base_freq / center_freq)
            impulse_samples = int(impulse_dur * self.SAMPLE_RATE)
            impulse_samples = max(10, impulse_samples)

            decay = random.uniform(*decay_range)
            decay = decay * (center_freq / base_freq)  # higher freqs decay faster

            # Generate the impulse
            impulse = self._generate_impulse(
                impulse_samples, center_freq,
                center_freq * 0.3, decay
            )

            # Random start position
            start = random.randint(0, max(0, total_samples - len(impulse) - 1))

            # Random amplitude
            amp = random.uniform(0.3, 1.0)
            # Favor quieter impulses (power-law distribution)
            amp = amp ** 2.0

            # Mix into output
            for j, val in enumerate(impulse):
                if start + j < total_samples:
                    output[start + j] += val * amp

        # Normalize
        max_val = max(abs(v) for v in output) if output else 1.0
        if max_val > 0:
            output = [v / max_val for v in output]

        return output

    def _ambient_tone(self, duration_sec, base_note_hz, chord_intervals, amp_envelope):
        """
        Generate a slow ambient musical tone that blends into the waves.
        Uses simple additive synthesis with slow amplitude modulation.
        """
        total_samples = int(duration_sec * self.SAMPLE_RATE)
        output = [0.0] * total_samples

        for interval in chord_intervals:
            freq = base_note_hz * (2 ** (interval / 12.0))
            # Slow LFO for organic movement
            lfo_rate = random.uniform(0.1, 0.4)
            lfo_depth = random.uniform(0.1, 0.3)

            for i in range(total_samples):
                t = i / self.SAMPLE_RATE
                # Amplitude envelope
                env = amp_envelope(t, duration_sec)
                # LFO modulation
                lfo = 1.0 + lfo_depth * math.sin(2 * math.pi * lfo_rate * t)
                # Slight pitch drift
                drift = 1.0 + 0.003 * math.sin(2 * math.pi * 0.05 * t + interval)
                phase = 2 * math.pi * freq * drift * t
                output[i] += env * lfo * math.sin(phase) * 0.3

        return output

    def generate(self, duration_sec=300, filename="seashore.wav"):
        """
        Generate the complete seashore WAV file.

        Layers:
          1. Deep rumble (sub-bass waves, 50-200 Hz)
          2. Mid waves (main ocean sound, 200-800 Hz)
          3. High splash (surface detail, 800-3000 Hz)
          4. Stereo spatialization
          5. Ambient music layer
        """
        print(f"🌊 Generating {duration_sec}s seashore (seed={self.seed})...")

        # ── Layer 1: Deep rumble (low frequency, long decay) ──────
        print("  Layer 1/5: deep rumble...")
        deep_rumble = self._layer_waves(
            num_impulses=200,
            duration_sec=duration_sec,
            base_freq=120,
            freq_spread=80,
            decay_range=(0.5, 2.0),
            density=1.0
        )

        # ── Layer 2: Mid waves (main body of ocean sound) ─────────
        print("  Layer 2/5: mid waves...")
        mid_waves = self._layer_waves(
            num_impulses=500,
            duration_sec=duration_sec,
            base_freq=400,
            freq_spread=300,
            decay_range=(2.0, 8.0),
            density=1.0
        )

        # ── Layer 3: Surface splash (high frequency detail) ───────
        print("  Layer 3/5: surface splash...")
        high_splash = self._layer_waves(
            num_impulses=1500,
            duration_sec=duration_sec,
            base_freq=1500,
            freq_spread=1200,
            decay_range=(5.0, 20.0),
            density=1.0
        )

        # ── Layer 4: Ambient music ────────────────────────────────
        print("  Layer 4/5: ambient tones...")
        # Random chord progression
        roots = [random.choice([55, 65.4, 73.4, 82.4, 98, 110])  # A2-C#3
                 for _ in range(random.randint(3, 5))]
        chord_types = [
            [0, 4, 7],      # major
            [0, 3, 7],      # minor
            [0, 5, 7],      # sus4
            [0, 4, 7, 11],  # maj7
            [0, 3, 7, 10],  # min7
        ]

        ambient = [0.0] * int(duration_sec * self.SAMPLE_RATE)
        section_len = duration_sec / len(roots)

        for idx, root in enumerate(roots):
            chord = random.choice(chord_types)
            start_t = idx * section_len
            end_t = min((idx + 1) * section_len, duration_sec)

            def envelope(t, total):
                # Fade in/out with crossfade between sections
                local_t = t - start_t
                dur = end_t - start_t
                if local_t < 0:
                    return 0.0
                # Gentle attack and release
                attack = min(local_t / 3.0, 1.0)  # 3-second attack
                release = min((dur - local_t) / 3.0, 1.0) if local_t < dur else 0.0
                return attack * release * 0.15  # keep tones subtle

            tone = self._ambient_tone(
                duration_sec=duration_sec,
                base_note_hz=root,
                chord_intervals=chord,
                amp_envelope=lambda t, d: envelope(t, d) * (
                    0.6 + 0.4 * math.sin(2 * math.pi * 0.03 * t)
                )
            )

            for i, v in enumerate(tone):
                if v != 0:
                    ambient[i] += v

        # ── Layer 5: Assemble and spatialize ──────────────────────
        print("  Layer 5/5: mixing and spatializing...")
        total_samples = int(duration_sec * self.SAMPLE_RATE)

        # Spatialization: offset left/right channels slightly
        # for stereo depth
        left_channel = [0.0] * total_samples
        right_channel = [0.0] * total_samples

        for i in range(total_samples):
            # Layer weights
            l1 = deep_rumble[i] * 1.0
            l2 = mid_waves[i] * 0.8
            l3 = high_splash[i] * 0.4
            l4 = ambient[i] * 1.0

            # Basic stereo: slightly different mixing per channel
            # Simulate spatial position through amplitude differences
            left = l1 * 0.7 + l2 * 1.0 + l3 * 1.2 + l4 * 0.5
            right = l1 * 1.0 + l2 * 0.7 + l3 * 0.8 + l4 * 0.8

            # Add slight decorrelation for wider stereo
            if i > 1:
                left += (high_splash[i-1] - high_splash[i]) * 0.2
                right -= (mid_waves[i-1] - mid_waves[i]) * 0.15

            left_channel[i] = left
            right_channel[i] = right

        # ── Normalize and write WAV ───────────────────────────────
        print(f"  Writing {filename}...")
        max_amp = 0.0
        for i in range(total_samples):
            max_amp = max(max_amp, abs(left_channel[i]), abs(right_channel[i]))

        scale = 0.85 / max_amp if max_amp > 0 else 1.0

        with wave.open(filename, 'w') as wf:
            wf.setnchannels(self.NUM_CHANNELS)
            wf.setsampwidth(self.BITS_PER_SAMPLE // 8)
            wf.setframerate(self.SAMPLE_RATE)
            wf.setnframes(total_samples)

            for i in range(total_samples):
                left_val = int(left_channel[i] * scale * 32767)
                right_val = int(right_channel[i] * scale * 32767)

                # Clamp
                left_val = max(-32768, min(32767, left_val))
                right_val = max(-32768, min(32767, right_val))

                wf.writeframes(struct.pack('<hh', left_val, right_val))

        # File size
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"  ✅ Done! {filename} ({size_mb:.1f} MB, {duration_sec}s)")

    @staticmethod
    def preview_stats(filename):
        """Print statistics about a generated WAV file."""
        with wave.open(filename, 'r') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            duration = frames / rate

            print(f"\n  📊 {filename}:")
            print(f"     Duration: {duration:.1f}s")
            print(f"     Sample rate: {rate} Hz")
            print(f"     Channels: {channels}")
            print(f"     Frames: {frames}")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Synthetic Seashore — procedural ocean wave WAV generator"
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=60,
        help="Duration in seconds (default: 60, for demo; full: 300)"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None,
        help="Random seed (default: random)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="seashore.wav",
        help="Output WAV filename"
    )

    args = parser.parse_args()

    gen = SeashoreGenerator(seed=args.seed)
    gen.generate(duration_sec=args.duration, filename=args.output)
    SeashoreGenerator.preview_stats(args.output)


if __name__ == "__main__":
    main()
