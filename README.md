<div align="center">

<img src="assets/og-preview.jpg" alt="Divya Drusti — Spatial Vision Studio by Razel Tech" width="50%">

# DIVYA DRUSTI — Spatial Vision Studio

**Real-time AR spatial vision and virtual instrument studio running entirely on-device in your browser.**  
Shader prisms, AI hair & makeup customization, Jutsu kinetic synthesis, Air Music instruments, and STEM educational games.  
No install required. Zero server uploads. Pure client-side WebGL2 and MediaPipe AI.

[![Powered by Razel Tech](https://img.shields.io/badge/Powered%20by-Razel%20Tech-3b82f6?style=flat-square&logo=github)](https://razeltech.github.io)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-10b981?style=flat-square)](https://razeltech.github.io)
[![PWA Ready](https://img.shields.io/badge/PWA-Offline%20Ready-f59e0b?style=flat-square)](https://razeltech.github.io)
[![License](https://img.shields.io/badge/License-MIT-e11d48?style=flat-square)](#credits)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Scenes](#feature-scenes)
3. [Air Music Suite](#air-music-suite)
4. [Kinetic Jutsu & Proximity VFX](#kinetic-jutsu--proximity-vfx)
5. [AR Makeup & Salon Customization](#ar-makeup--salon-customization)
6. [Spatial Prism & Multi-Modal Controls](#spatial-prism--multi-modal-controls)
7. [Prana Catch & Educational STEM Curriculum](#prana-catch--educational-stem-curriculum)
8. [Studio Tools & System Controls](#studio-tools--system-controls)
9. [PWA Offline Support & Direct Git Updates](#pwa-offline-support--direct-git-updates)
10. [Run Locally](#run-locally)
11. [Deploy on GitHub Pages](#deploy-on-github-pages)
12. [Technology](#technology)
13. [Browser Support](#browser-support)
14. [Credits](#credits)

---

## Overview

**Divya Drusti** (दिव्य दृष्टि — "Divine Vision" in Sanskrit) is a zero-build browser application that transforms any web camera into an augmented reality and spatial interaction studio. Powered by Google MediaPipe for on-device tracking (hands, face mesh, body pose, multiclass segmentation) and WebGL2 shader pipelines, it provides low-latency spatial computing with zero cloud dependence.

Everything executes locally in the client runtime. Video feeds remain completely private and never leave your hardware.

**Powered by [Razel Tech](https://razeltech.github.io)**

---

## Feature Scenes

| Scene | Core Capabilities |
|---|---|
| **Spatial Prism** | Floating quadrilateral viewport with 14 custom shaders (Spectral Prism, Liquid Bubble, Bendy Fabric, Point Cloud, Depth Color, Gaussian Splat, Thermal IR, Anime Cel, C-41 Negative). Supports single/dual pinch, 4-corner touch dragging, quad locking, and gesture swiping. |
| **Hair & Makeup AR** | Real-time multiclass hair segmentation and face mesh overlay. Customizable hair palettes, custom color picker, lip tint finishes, cheek blush AR on facial landmarks, and skin smoothing with dynamic slider fills. |
| **Mystic Jutsu** | Kinetic spellcasting scaled to hand openness and palm orientation: Eldritch Shield, Chidori Lightning, Rasengan Vortex, Sling Portal, Flow Ribbons, Procedural Vector Alchemy Circle, 3D Point Cloud, Finger Text, Light Painter, and Fish Flock Swarm. |
| **Air Music** | Polyphonic Air Piano (floating elevation modes, dynamic octave tracking), Virtual Holographic Drum Kit with debounced striking, and 6-string Acoustic Air Guitar with physical wave harmonics and chord selection. |
| **3D Avatar** | Humanoid motion-capture rig powered by Three.js and MediaPipe Pose retargeting. Live camera is replaced with a 3D character in real time on switchable backdrops. |
| **Prana Catch** | Physical reflex and STEM learning game. Catch falling answer orbs using head tilt, nose tracking, mouth opening, or index fingertip. Features collision-free auto-scaling question cards on mobile. |

---

## Air Music Suite

The Air Music engine provides three polyphonic virtual instruments running on the Web Audio API with zero external audio samples:

### 1. Polyphonic Finger Piano
- **Tracking & Interaction:** Tracks all 5 fingertips per hand (Thumb, Index, Middle, Ring, Pinky) plus multi-touch touchscreen and mouse cursors.
- **Dynamic Octave Labels:** Supports 3 distinct octave ranges (`C3–E4`, `C4–E5`, `C5–E6`). Key labels dynamically update to match the active octave.
- **Adjustable Floating Elevation:** Toggle between Floating (`44% height`), Desk (`62% height`), and High (`30% height`) via `#btn-music-level`.
- **Synthesis Model:** 3-harmonic sine synthesis with fast 6ms attack and 380ms exponential decay.

### 2. Holographic Drum Kit
- **Pads:** 7 radial pads (Crash, Hi-Hat, High Tom, Ride, Snare, Floor Tom, Bass Kick).
- **Physical Debouncing:** Independent touch-state debounce tracking prevents accidental multi-triggering while allowing rapid rolls.
- **Audio Models:** Synthesized analog kick with pitch decay, dual-layer snare with white noise rattle, filtered hi-hats, and resonant tom oscillators.

### 3. Acoustic Air Guitar
- **Layout:** Horizontal 6-string fretboard with left-side chord selector panel (`OPEN`, `Em`, `G`, `C`, `D`, `Am`, `F`) and right-side strum zone.
- **Physical String Synthesis:** Dual oscillator (Triangle + Sawtooth) passing through an exponential resonant lowpass biquad filter ($Q = 3.5$) with natural acoustic body decay.
- **Oscillating Wave Harmonics:** Plucked strings vibrate with real sinusoidal wave physics on Canvas 2D, damped over time.

---

## Kinetic Jutsu & Proximity VFX

### Next-Level Two-Hand Proximity Dynamics
When both hands enter the frame, Divya Drusti computes a continuous proximity factor ($0.0 \to 1.0$) driving high-energy VFX:
- **Braided Quantum Plasma Filaments:** 5 undulating plasma ribbons linking opposing fingertips (Thumb, Index, Middle, Ring, Pinky).
- **Dipolar Magnetic Forcefield Arcs:** High-voltage arcs and magnetic field lines spanning between both palms.
- **Harmonic Shockwave Rings:** Expanding mid-point interference rings that intensify as hands converge.

### Proximity Clash & Fusion States (Distance < 180px)
- **Chidori Hyper-Singularity Clash:** Blinding white central core with 8 branching high-voltage discharges and electric arcs.
- **Aegis Sacred Bastion:** Dual counter-rotating 12-point Merkaba stars with radial celestial god-rays.
- **Oodama Chakra Vortex:** 6 multi-axis tilted orbital accretion discs with high-density plasma particles.
- **Wormhole Singularity Rift:** 4-armed golden logarithmic spiral accretion funnel.

### Procedural Vector Alchemical Sigils
To ensure zero missing glyphs, tofu boxes, or blank white circles across mobile devices, the Alchemy Circle renders 12 procedural vector sigils via Canvas 2D paths:
1. Sun (Outer ring with central solar core)
2. Moon (Crescent)
3. Mercury (Circle with crescent horns and base cross)
4. Venus (Circle with base cross)
5. Mars (Circle with diagonal arrow)
6. Jupiter (Cross with curved harmonic loop)
7. Saturn (Cross with sickle foot)
8. Fire (Upward equilateral triangle)
9. Water (Downward equilateral triangle)
10. Air (Upward triangle with horizontal crossbar)
11. Earth (Downward triangle with horizontal crossbar)
12. Ouroboros (Infinity figure-8 loop)

---

## AR Makeup & Salon Customization

- **Hair Dye:** 6 curated color presets plus a full 24-bit custom RGB color picker (`#picker-hair-color`). Preserves hair specularity, ambient lighting, and natural depth.
- **Lipstick:** 5 satin/velvet swatches plus custom color picker (`#picker-lip-color`) and opacity slider.
- **Cheek Blush AR:** Natural radial cheek blush mapped directly to facial landmarks 205 and 425 with Peach Bloom, Petal Rose, Warm Coral, Berry Wine, and Golden Sunset presets, custom color picker, and dynamic opacity slider (`#slider-blush-opacity`).
- **Dynamic Slider Fills:** All sliders dynamically compute active blue linear-gradient fill tracks matching the current value.

---

## Spatial Prism & Multi-Modal Controls

- **Single & Dual Pinch:** Pinch with both hands to expand/contract the spatial prism; single-hand pinch supported on mobile.
- **Touch Corner Dragging:** 4 interactive corner handles with 55px touch targets for direct manipulation.
- **Quad Lock:** `#btn-prism-lock` freezes the prism geometry in place while keeping shader effects active.
- **Filter Navigation:**
  - **Hand Swipe Gesture:** Swipe your hand rapidly left or right across the camera to cycle through shader filters (with 650ms debounce and velocity thresholding).
  - **Prev / Next Dock Controls:** `#btn-prism-prev` and `#btn-prism-next` buttons in the prism dock.
  - **Wink Detection:** Hands-free filter cycling by winking either eye.

---

## Prana Catch & Educational STEM Curriculum

- **Collision-Free Mobile Question Card:** Question boxes are positioned at `y = 104px`, safely below the top status badge. Question text dynamically auto-scales (13px–20px) and wraps into multi-line layouts so text never spills out of the card.
- **Dedicated Feedback Pill:** Correct/incorrect notifications display in a distinct pill container below the card.
- **STEM Presets:** Math, Physics, Chemistry, Biology, Astronomy, World Capitals, and custom CSV teacher pack loader.

---

## Studio Tools & System Controls

- **Manual Snapshot & Freeze-Frame Modal:** Capture high-resolution PNG snapshots with branded watermark via HUD button, Settings drawer, or `Space`/`C` keys. Opens a freeze-frame modal with Save Image and Retake options.
- **Camera Switch & Auto-Mirror:** Switch between front and rear cameras. Mirror mode automatically disables on environment cameras and displays clean SVG reflections.
- **Screen Wake Lock:** Prevents screen dimming or sleeping during active AR sessions; auto-reacquires when returning to the browser tab.
- **Spotlight Walkthrough:** Guided first-time tour that saves completion state to localStorage, never intruding on returning sessions.

---

## PWA Offline Support & Direct Git Updates

- **Offline Execution:** The Service Worker (`sw.js`) caches the app shell and runtime MediaPipe WASM and model files (`storage.googleapis.com`, `cdn.jsdelivr.net`). Works 100% offline after the initial visit.
- **Standalone PWA Guard:** Automatically detects standalone APK / WebAPK mode (`display-mode: standalone`, `android-app://`) and suppresses the install banner and install drawer row.
- **Direct Git Update Checker:** The "Update / Git Sync" button in the Settings drawer checks `sw.update()` and fetches latest network commits, prompting an instant reload when an update is available.

---

## Run Locally

Camera access requires **HTTPS or localhost**.

```bash
npx serve .
# or
python3 -m http.server 8080
# Then navigate to http://localhost:8080
```

---

## Deploy on GitHub Pages

1. Push the repository to GitHub (`index.html`, `sw.js`, `manifest.json`, `css/`, `js/`, `assets/`, `icons/`).
2. Navigate to **Settings → Pages → Deploy from branch → `main` / root**.
3. Access `https://<username>.github.io/<repo>/` over HTTPS.

---

## Technology

| Layer | Architecture |
|---|---|
| **Shader Pipeline** | WebGL2 fragment shaders (14 custom visual filters) |
| **2D Overlays & Instruments** | HTML5 Canvas 2D API with sub-pixel vector rendering |
| **3D Avatar Engine** | Three.js with procedural bone retargeting |
| **Sound Synthesis** | Web Audio API (polyphonic additive, noise generators, biquad filters) |
| **AI Computer Vision** | MediaPipe Tasks Vision (Face Landmarker, Gesture Recognizer, Pose Landmarker, Image Segmenter) |
| **App Architecture** | Vanilla ES Modules, zero compilation, offline PWA Service Worker |

---

## Browser Support

Supported across Chrome, Edge, Safari 16+, and Firefox with WebGL2 and camera permissions enabled. Optimized for mobile touchscreens, tablets, laptops, and desktop workstations.

---

## Credits

Developed by **[Razel Tech](https://razeltech.github.io)**.  
Google MediaPipe (c) Google, Apache-2.0.

