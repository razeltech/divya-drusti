    import {
      FilesetResolver,
      GestureRecognizer,
      FaceLandmarker,
      PoseLandmarker,
      ImageSegmenter
    } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

    /* =========================================================================
       1. GLOBAL STATE & SOUND SYNTHESIS
       ========================================================================= */
    const CONFIG = Object.freeze({
      WASM_PATH: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
      MODELS: {
        GESTURE: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        FACE: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        POSE: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        SEGMENTER: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
      }
    });

    let lastHandSwipeTime = 0;
    let prevHandX = null;
    let prevHandTime = 0;

    const state = {
      scene: "PRISM", // "PRISM" | "MAKEUP" | "MYSTIC" | "GAME"
      filter: "SPECTRAL_PRISM",
      spell: "SHIELD",
      
      // Overlays default OFF
      showHands: false,
      showPose: false,
      showFaceMesh: false,

      prismActive: false,
      prismOpacity: 0.0,
      prismLocked: false,
      mirrorView: true,
      hasCamera: false,
      soundEnabled: true,

      // Maximum Quality Parameters
      density: 1.0,
      splatRadius: 1.0,
      edgeGain: 1.0,
      vibrance: 0.85,

      // Hair & Makeup
      hairShade: "rosegold",
      lipColor: "#e11d48",

      // Prana Game
      gameScore: 0,
      gems: [],
      particles: [],

      // Mystic Jutsu State
      mandalaAngle: 0,
      lightningBranches: [],
      portalSparks: [],

      availableCameras: [],
      currentCameraIndex: 0,

      // Spatial Prism Corners (Normalized [0, 1])
      corners: {
        tl: { x: 0.25, y: 0.25 },
        tr: { x: 0.75, y: 0.25 },
        br: { x: 0.75, y: 0.75 },
        bl: { x: 0.25, y: 0.75 }
      },
      targetCorners: {
        tl: { x: 0.25, y: 0.25 },
        tr: { x: 0.75, y: 0.25 },
        br: { x: 0.75, y: 0.75 },
        bl: { x: 0.25, y: 0.75 }
      },

      handData: [],
      poseLandmarks: null,
      faceLandmarks: null,
      jawOpen: 0,
      lastWinkTime: 0
    };

    // Polyphonic Web Audio Engine
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    function playAudioTone(freq = 440, type = "sine", duration = 0.25, gainVal = 0.2) {
      if (!state.soundEnabled) return;
      if (audioCtx.state === "suspended") audioCtx.resume();
      try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(gainVal, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration + 0.02);
      } catch (e) {}
    }

    /* =========================================================================
       2. OPTICAL FIT (LUSTRA 100% ZERO-ZOOM MATHEMATICS)
       ========================================================================= */
    const frame = document.getElementById("viewport-frame");
    const glCanvas = document.getElementById("render-surface");
    const vecCanvas = document.getElementById("vector-overlay");
    const video = document.getElementById("webcam-source");

    // Offscreen hair mask canvases
    const hairCanvas = document.createElement("canvas");
    const hairCtx = hairCanvas.getContext("2d");
    const hairLayer = document.createElement("canvas");
    const hairLayerCtx = hairLayer.getContext("2d");

    function fitViewport() {
      const container = document.getElementById("stage-container");
      const stageW = container.clientWidth;
      const stageH = container.clientHeight;

      const vw = (state.hasCamera && video.videoWidth) ? video.videoWidth : 1280;
      const vh = (state.hasCamera && video.videoHeight) ? video.videoHeight : 720;

      // Lustra uncropped natural optical fit
      const k = Math.min(stageW / vw, stageH / vh);
      const targetW = Math.round(vw * k);
      const targetH = Math.round(vh * k);

      frame.style.width = targetW + "px";
      frame.style.height = targetH + "px";

      glCanvas.width = targetW;
      glCanvas.height = targetH;
      vecCanvas.width = targetW;
      vecCanvas.height = targetH;

      hairCanvas.width = targetW;
      hairCanvas.height = targetH;
      hairLayer.width = targetW;
      hairLayer.height = targetH;

      if (gl) gl.viewport(0, 0, targetW, targetH);
    }

    window.addEventListener("resize", fitViewport);
    screen.orientation?.addEventListener("change", () => setTimeout(fitViewport, 200));

    function landmarkToScreen(lm) {
      const cw = vecCanvas.width;
      const ch = vecCanvas.height;
      const normX = state.mirrorView ? (1.0 - lm.x) : lm.x;
      const normY = lm.y;

      return {
        x: normX * cw,
        y: normY * ch,
        normX: normX,
        normY: normY,
        z: lm.z || 0
      };
    }

    /* =========================================================================
       3. WEBGL2 SHADER PIPELINE (ALL 8 RESTORED REALITY FILTERS)
       ========================================================================= */
    const gl = glCanvas.getContext("webgl2", { antialias: true, alpha: false });

    const VS_SOURCE = `#version 300 es
      in vec2 a_position;
      out vec2 v_screenUv;

      void main() {
        v_screenUv = vec2((a_position.x + 1.0) * 0.5, (1.0 - a_position.y) * 0.5);
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;

    const FS_SOURCE = `#version 300 es
      precision highp float;
      in vec2 v_screenUv;
      out vec4 fragColor;

      uniform sampler2D u_video;
      uniform vec2 u_resolution;
      uniform int u_mirror;
      uniform int u_filter; // 0..7
      uniform float u_prismAlpha;
      uniform vec2 u_v0;
      uniform vec2 u_v1;
      uniform vec2 u_v2;
      uniform vec2 u_v3;

      uniform float u_density;
      uniform float u_splatRadius;
      uniform float u_edgeGain;
      uniform float u_time;
      uniform float u_zoom;
      uniform vec2 u_pointer;
      uniform float u_pointerOn;
      uniform sampler2D u_mask;
      uniform sampler2D u_mk;
      uniform int u_makeup;
      uniform vec3 u_hairA;
      uniform vec3 u_hairB;
      uniform vec3 u_hairC;
      uniform float u_hairOn;
      uniform float u_vibrance;
      uniform float u_smooth;
      float hash11(float n) { return fract(sin(n * 127.1) * 43758.5453); }
      float hash21(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
      vec3 turbo(float x) {
        x = clamp(x, 0.0, 1.0);
        vec4 v4 = vec4(1.0, x, x * x, x * x * x);
        vec2 v2 = v4.zw * v4.z;
        return vec3(
          dot(v4, vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2(-152.94239396, 59.28637943)),
          dot(v4, vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2(4.27729857, 2.82956604)),
          dot(v4, vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2(-89.90310912, 27.34824973)));
      }

      // Robust quad test: two triangles (works for crossed / non-convex quads, any winding),
      // zero-area (collapsed "pinpoint") quads are treated as EMPTY instead of filling the screen.
      float triArea2(vec2 a, vec2 b, vec2 c) { return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y); }
      bool inTri(vec2 p, vec2 a, vec2 b, vec2 c) {
        if (abs(triArea2(a, b, c)) < 1e-6) return false;
        float d1 = triArea2(p, a, b), d2 = triArea2(p, b, c), d3 = triArea2(p, c, a);
        bool neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        bool pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
        return !(neg && pos);
      }
      float sdSeg(vec2 p, vec2 a, vec2 b) {
        vec2 pa = p - a, ba = b - a;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
        return length(pa - ba * h);
      }

      // Even-odd outline test: exactly the area enclosed by the 4 edges, even when twisted into a bow-tie
      bool edgeX(vec2 p, vec2 s, vec2 e) {
        return ((s.y > p.y) != (e.y > p.y)) && (p.x < (e.x - s.x) * (p.y - s.y) / (e.y - s.y + 1e-9) + s.x);
      }
      bool inQuadEO(vec2 p, vec2 a, vec2 b, vec2 c, vec2 d) {
        bool r = false;
        if (edgeX(p, a, b)) r = !r;
        if (edgeX(p, b, c)) r = !r;
        if (edgeX(p, c, d)) r = !r;
        if (edgeX(p, d, a)) r = !r;
        return r;
      }

      vec2 getCameraUv(vec2 screenUv) {
        vec2 uv = screenUv;
        if (u_mirror == 1) uv.x = 1.0 - uv.x;
        return uv;
      }

      float getLuminance(vec3 c) {
        return dot(c, vec3(0.299, 0.587, 0.114));
      }

      float getSobel(vec2 uv, vec2 step) {
        float t00 = getLuminance(texture(u_video, getCameraUv(uv + vec2(-step.x, -step.y))).rgb);
        float t01 = getLuminance(texture(u_video, getCameraUv(uv + vec2(0.0, -step.y))).rgb);
        float t02 = getLuminance(texture(u_video, getCameraUv(uv + vec2(step.x, -step.y))).rgb);
        float t10 = getLuminance(texture(u_video, getCameraUv(uv + vec2(-step.x, 0.0))).rgb);
        float t12 = getLuminance(texture(u_video, getCameraUv(uv + vec2(step.x, 0.0))).rgb);
        float t20 = getLuminance(texture(u_video, getCameraUv(uv + vec2(-step.x, step.y))).rgb);
        float t21 = getLuminance(texture(u_video, getCameraUv(uv + vec2(0.0, step.y))).rgb);
        float t22 = getLuminance(texture(u_video, getCameraUv(uv + vec2(step.x, step.y))).rgb);

        float gx = -t00 - 2.0*t10 - t20 + t02 + 2.0*t12 + t22;
        float gy = -t00 - 2.0*t01 - t02 + t20 + 2.0*t21 + t22;
        return sqrt(gx*gx + gy*gy) * u_edgeGain;
      }

      vec3 getThermal(float t) {
        vec3 c0 = vec3(0.05, 0.05, 0.18);
        vec3 c1 = vec3(0.55, 0.05, 0.50);
        vec3 c2 = vec3(0.92, 0.28, 0.06);
        vec3 c3 = vec3(1.00, 0.88, 0.16);
        vec3 c4 = vec3(1.00, 1.00, 1.00);

        if (t < 0.25) return mix(c0, c1, t / 0.25);
        if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
        if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
        return mix(c3, c4, (t - 0.75) / 0.25);
      }

      void main() {
        vec2 texel = 1.0 / u_resolution;
        vec2 camUv = getCameraUv(v_screenUv);
        vec4 rawColor = texture(u_video, camUv);

        // ---- CLEAN MAKEUP BLEND (luminance-preserving, in-shader) ----
        if (u_makeup == 1) {
          vec3 base = rawColor.rgb;
          vec2 mp = vec2(1.5 / 256.0);
          vec2 m = texture(u_mask, camUv).rg * 2.0
                 + texture(u_mask, camUv + vec2(mp.x, 0.0)).rg + texture(u_mask, camUv - vec2(mp.x, 0.0)).rg
                 + texture(u_mask, camUv + vec2(0.0, mp.y)).rg + texture(u_mask, camUv - vec2(0.0, mp.y)).rg;
          m /= 6.0;
          float skinM = smoothstep(0.35, 0.8, m.g);
          float hairM = smoothstep(0.3, 0.75, m.r) * u_hairOn;
          if (skinM > 0.02 && u_smooth > 0.01) {
            vec3 acc = vec3(0.0); float ws = 0.0;
            for (int i = -2; i <= 2; i++) for (int j = -2; j <= 2; j++) {
              vec3 s = texture(u_video, camUv + vec2(float(i), float(j)) * texel * 2.2).rgb;
              vec3 d = s - base;
              float wgt = exp(-dot(d, d) * 45.0);
              acc += s * wgt; ws += wgt;
            }
            base = mix(base, (acc / ws) * 1.03, skinM * u_smooth);
          }
          if (hairM > 0.01) {
            float lum = getLuminance(base);
            float g = smoothstep(0.1, 0.9, v_screenUv.y);
            vec3 tint = g < 0.5 ? mix(u_hairA, u_hairB, g * 2.0) : mix(u_hairB, u_hairC, (g - 0.5) * 2.0);
            vec3 dyed = tint * (0.22 + 1.4 * pow(lum, 0.85)) + vec3(pow(lum, 5.0)) * 0.35;
            base = mix(base, clamp(dyed, 0.0, 1.0), hairM * u_vibrance);
          }
          vec4 mk = texture(u_mk, v_screenUv);
          base = mix(base, mk.rgb * (0.55 + 0.9 * getLuminance(base)), mk.a);
          rawColor = vec4(clamp(base, 0.0, 1.0), 1.0);
        }

        if (u_prismAlpha <= 0.001) {
          fragColor = rawColor;
          return;
        }

        vec2 asp2 = vec2(u_resolution.x / u_resolution.y, 1.0);
        vec2 P = v_screenUv * asp2, Q0 = u_v0 * asp2, Q1 = u_v1 * asp2, Q2 = u_v2 * asp2, Q3 = u_v3 * asp2;
        bool inPrism = inQuadEO(P, Q0, Q1, Q2, Q3);
        bool quadOk = abs(triArea2(Q0, Q1, Q2)) + abs(triArea2(Q0, Q2, Q3)) > 2e-6;
        float dEdge = min(min(sdSeg(P, Q0, Q1), sdSeg(P, Q1, Q2)), min(sdSeg(P, Q2, Q3), sdSeg(P, Q3, Q0)));
        vec3 fx = rawColor.rgb;

        if (u_filter == 0) {
          // 0: SPECTRAL PRISM (High-Performance Cauchy Dispersion & Prismatic Caustics)
          vec2 center = (u_v0 + u_v1 + u_v2 + u_v3) * 0.25;
          vec2 toCenter = v_screenUv - center;
          vec2 toCenterAsp = toCenter * asp2;
          float distCenter = length(toCenterAsp);
          vec2 refrDir = distCenter > 1e-4 ? normalize(toCenterAsp) / asp2 : vec2(0.0, 1.0);

          // Diagonal facet geometry (cut crystal prism facets)
          vec2 uvd = v_screenUv - center;
          float facetX = uvd.x * 0.7071 - uvd.y * 0.7071;
          float facetY = uvd.x * 0.7071 + uvd.y * 0.7071;
          vec2 facetNormal = normalize(vec2(sign(facetX) * 0.35, sign(facetY) * 0.35) + toCenter * 1.4 + 1e-4);

          // TouchDesigner interactive hand ripple
          float ripple = 1.0;
          if (u_pointerOn > 0.5) {
            float pDist = length((v_screenUv - u_pointer) * asp2);
            ripple += 2.8 * sin(pDist * 32.0 - u_time * 5.0) * exp(-pDist * 7.0);
          }

          // Edge bevel refraction normal
          float bevelWidth = 0.08;
          float bevelDist = clamp(dEdge / bevelWidth, 0.0, 1.0);
          float bevelCurv = 1.0 - sqrt(bevelDist);
          vec2 refrVec = mix(facetNormal * 0.022, refrDir * 0.042, 0.65) * ripple;
          refrVec += refrDir * (bevelCurv * 0.038);

          // Base Cauchy dispersion magnitude
          float dispBase = (0.016 * u_splatRadius + distCenter * 0.026 + bevelCurv * 0.04) * ripple;

          // High-Efficiency 3-Channel Dispersion (Locked 60 FPS on mobile GPUs)
          vec3 refracted;
          refracted.r = texture(u_video, getCameraUv(v_screenUv - refrVec - refrDir * (dispBase * 1.35))).r;
          refracted.g = texture(u_video, getCameraUv(v_screenUv - refrVec)).g;
          refracted.b = texture(u_video, getCameraUv(v_screenUv - refrVec + refrDir * (dispBase * 1.35))).b;

          // Prismatic Internal Caustics & Sweeping Rainbow Flares
          float facetAngle = atan(toCenterAsp.y, toCenterAsp.x);
          float causticPattern = sin(dot(toCenterAsp, vec2(cos(u_time * 0.4), sin(u_time * 0.4))) * 38.0
                                    + sin(facetAngle * 5.0 + u_time * 0.5) * 3.5);
          float causticRays = pow(abs(causticPattern), 6.0) * (0.6 + 0.4 * sin(u_time * 2.0));
          vec3 causticColor = 0.5 + 0.5 * cos(6.28318 * (vec3(0.0, 0.33, 0.67) + facetAngle * 0.4 + u_time * 0.22));

          // Crystal Specular Sheen (sweeping light bar across glass pane)
          float lightSweep = dot(v_screenUv - vec2(0.5), vec2(0.7071, 0.7071));
          float glint = pow(max(0.0, sin(lightSweep * 14.0 - u_time * 1.8 + sin(distCenter * 10.0))), 16.0);

          // Total Internal Reflection (TIR) Rim Lighting
          float rimGlow = smoothstep(0.08, 0.005, dEdge) * u_edgeGain;

          fx = refracted * (0.94 + 0.22 * causticRays)
             + causticColor * (causticRays * 0.55 * u_splatRadius + rimGlow * 0.6)
             + vec3(glint * 0.45 + rimGlow * 0.38);

        } else if (u_filter == 1) {
          // 1: BENDY HOLOGRAPHIC FABRIC (TouchDesigner Wavy Iridescent Refraction)
          vec2 wave = vec2(
            sin(v_screenUv.y * 22.0 + u_time * 3.2) * 0.016 + cos(v_screenUv.x * 14.0 - u_time * 2.0) * 0.012,
            cos(v_screenUv.x * 22.0 - u_time * 2.8) * 0.016 + sin(v_screenUv.y * 14.0 + u_time * 2.4) * 0.012
          );
          vec2 wUv = v_screenUv + wave;
          vec3 cR = texture(u_video, getCameraUv(wUv + vec2(0.012, 0.0))).rgb;
          vec3 cG = texture(u_video, getCameraUv(wUv)).rgb;
          vec3 cB = texture(u_video, getCameraUv(wUv - vec2(0.012, 0.0))).rgb;
          vec3 chroma = vec3(cR.r, cG.g, cB.b);
          float lum = getLuminance(chroma);
          vec3 irid = 0.5 + 0.5 * cos(6.28318 * (lum * 1.8 + vec3(0.0, 0.33, 0.67) + u_time * 0.25));
          float spec = pow(max(0.0, sin(v_screenUv.y * 18.0 + u_time * 3.5 + v_screenUv.x * 8.0)), 6.0);
          fx = mix(chroma, irid * 1.35, 0.42) + vec3(spec * 0.45);

        } else if (u_filter == 2) {
          // 2: STRETCHY LIQUID BUBBLE (TouchDesigner Iridescent Bubble Membrane)
          vec2 center = (Q0 + Q1 + Q2 + Q3) * 0.25;
          vec2 delta = P - center;
          float dist = length(delta);
          float wobble = sin(dist * 18.0 - u_time * 3.8) * 0.02;
          vec2 bUv = v_screenUv + normalize(delta + vec2(1e-4)) * wobble;
          vec3 bR = texture(u_video, getCameraUv(bUv + vec2(0.016, 0.005))).rgb;
          vec3 bG = texture(u_video, getCameraUv(bUv)).rgb;
          vec3 bB = texture(u_video, getCameraUv(bUv - vec2(0.016, 0.005))).rgb;
          vec3 bubbleCol = vec3(bR.r, bG.g, bB.b);
          float rim = smoothstep(0.0, 0.35, dist);
          vec3 thinFilm = 0.5 + 0.5 * cos(6.28318 * (dist * 2.5 + vec3(0.1, 0.45, 0.8) + u_time * 0.2));
          fx = mix(bubbleCol, thinFilm * 1.4, rim * 0.55);

        } else if (u_filter == 3) {
          // 3: PROXIMITY POP / PLEXUS PASS-THROUGH
          fx = rawColor.rgb;

        } else if (u_filter == 4) {
          // 4: PRISM CRISSCROSS (two diagonal glass lattices, chromatic dispersion, hand ripple)
          float asp = u_resolution.x / u_resolution.y;
          float sc = 8.0 + 16.0 * u_density;
          vec2 q = v_screenUv * vec2(asp, 1.0) * sc;
          float da = q.x + q.y, db = q.x - q.y;
          float ia = floor(da), ib = floor(db);
          float fa = fract(da), fb = fract(db);
          vec2 dirA = vec2(0.7071, 0.7071), dirB = vec2(0.7071, -0.7071);
          float ripple = 1.0;
          if (u_pointerOn > 0.5) ripple += 2.5 * exp(-length((v_screenUv - u_pointer) * vec2(asp, 1.0)) * 5.0);
          vec2 off = (dirA * (hash11(ia) - 0.5) + dirB * (hash11(ib + 31.0) - 0.5)) * 0.04 * u_splatRadius * ripple;
          vec3 col;
          col.r = texture(u_video, getCameraUv(v_screenUv + off * 0.8)).r;
          col.g = texture(u_video, getCameraUv(v_screenUv + off * 1.0)).g;
          col.b = texture(u_video, getCameraUv(v_screenUv + off * 1.3)).b;
          float eA = smoothstep(0.07, 0.0, min(fa, 1.0 - fa));
          float eB = smoothstep(0.07, 0.0, min(fb, 1.0 - fb));
          vec3 rainbow = 0.5 + 0.5 * cos(6.2831 * (vec3(0.0, 0.33, 0.67) + (da + db) * 0.015 + u_time * 0.2));
          float facet = 0.82 + 0.35 * hash11(ia * 7.0 + ib);
          fx = col * facet + rainbow * max(eA, eB) * 0.5 * u_edgeGain + vec3(eA * eB) * 0.7;

        } else if (u_filter == 5) {
          // 2: KALEIDOSCOPE (centre follows index finger, 6-fold optical symmetry)
          float asp = u_resolution.x / u_resolution.y;
          vec2 c = u_pointerOn > 0.5 ? u_pointer : vec2(0.5);
          vec2 p = (v_screenUv - c) * vec2(asp, 1.0);
          float r = length(p);
          float seg = 6.2831853 / 6.0;
          float a = abs(mod(atan(p.y, p.x) + u_time * 0.12, seg) - seg * 0.5);
          vec2 q = vec2(cos(a), sin(a)) * r;
          q.x /= asp;
          vec2 uv = 1.0 - abs(1.0 - mod(q + c, 2.0));
          vec3 k;
          k.r = texture(u_video, getCameraUv(uv * 1.012 - 0.006)).r;
          k.g = texture(u_video, getCameraUv(uv)).g;
          k.b = texture(u_video, getCameraUv(uv * 0.988 + 0.006)).b;
          fx = k * 1.12;

        } else if (u_filter == 6) {
          // 3: ZOOM LENS (magnifies the area inside the prism, slight lens fringe)
          vec2 c = (u_v0 + u_v1 + u_v2 + u_v3) * 0.25;
          vec2 d = v_screenUv - c;
          float z = max(u_zoom, 1.0);
          vec3 zc;
          zc.r = texture(u_video, getCameraUv(clamp(c + d / (z * 1.012), 0.0, 1.0))).r;
          zc.g = texture(u_video, getCameraUv(clamp(c + d / z, 0.0, 1.0))).g;
          zc.b = texture(u_video, getCameraUv(clamp(c + d / (z * 0.988), 0.0, 1.0))).b;
          fx = zc;

        } else if (u_filter == 7) {
          // 4: NEON EDGE (hue-cycling edge glow)
          float lum = getLuminance(rawColor.rgb);
          float e1 = getSobel(v_screenUv, texel * 1.5);
          float e2 = getSobel(v_screenUv, texel * 5.0);
          vec3 hue = 0.5 + 0.5 * cos(6.2831 * (vec3(0.0, 0.33, 0.67) + lum * 0.6 + v_screenUv.x * 0.3 + u_time * 0.15));
          fx = hue * clamp(e1 * 2.2, 0.0, 1.0) * 1.5 + hue * e2 * 0.6 + rawColor.rgb * 0.1;

        } else if (u_filter == 8) {
          // 5: GAUSSIAN SPLATS (Radiant spherical falloff)
          float density = 50.0 * u_density;
          vec2 gd = vec2(density * u_resolution.x / u_resolution.y, density);
          vec2 cell = floor(v_screenUv * gd);
          vec2 center = (cell + 0.5) / gd;
          vec3 centerColor = texture(u_video, getCameraUv(center)).rgb;
          float lum = getLuminance(centerColor);

          vec2 coord = fract(v_screenUv * gd) - 0.5;
          float d = length(coord) * 2.0;
          float r = max(0.20, lum * 1.5 * u_splatRadius);
          float alpha = exp(-4.2 * (d / r) * (d / r));
          fx = mix(vec3(0.03, 0.04, 0.06), centerColor * 1.45, clamp(alpha, 0.0, 1.0));

        } else if (u_filter == 9) {
          // 6: TOPOLOGICAL MESH / POINT CLOUD (Elevation isolines)
          float grid = 64.0 * u_density;
          vec2 gg = vec2(grid * u_resolution.x / u_resolution.y, grid);
          vec2 pCell = floor(v_screenUv * gg);
          vec2 pCenter = (pCell + 0.5) / gg;
          vec3 pColor = texture(u_video, getCameraUv(pCenter)).rgb;
          float lum = getLuminance(pColor);

          vec2 delta = fract(v_screenUv * gg) - 0.5;
          float dotDist = length(delta) * 2.0;
          float ring = abs(sin(lum * 18.0 - dotDist * 3.0));
          vec3 isoline = mix(vec3(0.1, 0.4, 0.9), vec3(0.9, 0.2, 0.5), lum);
          fx = mix(vec3(0.04, 0.05, 0.08), isoline * 1.3, smoothstep(0.4, 0.1, dotDist) + ring * 0.35);

        } else if (u_filter == 10) {
          // 7: DEPTH COLOR (AI person mask + scene gradient -> turbo depth map with contour lines)
          vec2 mp = vec2(2.0 / 256.0);
          vec3 mm = (texture(u_mask, camUv).rgb * 2.0 + texture(u_mask, camUv + vec2(mp.x, 0.0)).rgb + texture(u_mask, camUv - vec2(mp.x, 0.0)).rgb
                   + texture(u_mask, camUv + vec2(0.0, mp.y)).rgb + texture(u_mask, camUv - vec2(0.0, mp.y)).rgb) / 6.0;
          float person = smoothstep(0.35, 0.75, mm.b);
          float lum = getLuminance(rawColor.rgb);
          float bg = 0.55 + 0.35 * (1.0 - v_screenUv.y) + 0.06 * (1.0 - lum);
          float fg = 0.08 + 0.2 * (1.0 - mm.g) + 0.08 * (1.0 - lum);
          float depth = mix(bg, fg, person);
          vec3 dc = turbo(1.0 - depth);
          float band = abs(fract(depth * 14.0) - 0.5);
          fx = mix(dc, rawColor.rgb, 0.18) * (1.0 - (1.0 - smoothstep(0.0, 0.04, band)) * 0.25);

        } else if (u_filter == 11) {
          // 8: QUANTUM GLITCH (RGB Aberration & CRT Scanlines)
          vec2 off = vec2(texel.x * 14.0, 0.0);
          float r = texture(u_video, getCameraUv(v_screenUv + off)).r;
          float g = texture(u_video, getCameraUv(v_screenUv)).g;
          float b = texture(u_video, getCameraUv(v_screenUv - off)).b;
          float scan = mod(gl_FragCoord.y, 3.0) < 1.0 ? 0.82 : 1.0;
          fx = vec3(r, g, b) * scan;

        } else if (u_filter == 12) {
          // 9: CYBER X-RAY (Fluoroscopic Radiometry)
          float lum = getLuminance(rawColor.rgb);
          float edge = getSobel(v_screenUv, texel);
          vec3 xray = vec3(1.0 - lum) * vec3(0.55, 0.80, 0.98) + vec3(edge * 1.9);
          fx = clamp(xray, 0.0, 1.0);

        } else if (u_filter == 13) {
          // 10: HOLO MATRIX (Phosphor Matrix Grid)
          float lum = getLuminance(rawColor.rgb);
          float grid = step(0.92, fract(v_screenUv.x * 80.0 * u_resolution.x / u_resolution.y)) + step(0.92, fract(v_screenUv.y * 80.0));
          vec3 phosphor = vec3(0.12, 0.95, 0.65) * lum;
          fx = mix(phosphor, vec3(0.85, 1.0, 0.9), grid * 0.4);

        } else if (u_filter == 14) {
          // 11: THERMAL IR (Radiometric false-color heat map)
          float lum = getLuminance(rawColor.rgb);
          float scanline = mod(gl_FragCoord.y, 4.0) < 1.0 ? 0.90 : 1.0;
          fx = getThermal(lum) * scanline;

        } else if (u_filter == 15) {
          // 12: ANIME CEL SHADING
          float edge = getSobel(v_screenUv, texel);
          float lum = getLuminance(rawColor.rgb);
          float qLum = floor(lum * 4.0) / 4.0;
          vec3 anime = mix(vec3(0.18, 0.18, 0.24), vec3(0.95, 0.88, 0.80), qLum);
          anime = mix(anime, rawColor.rgb, 0.45);
          fx = edge > 0.22 ? vec3(0.05, 0.06, 0.08) : anime;

        } else {
          // 13: C-41 FILM NEGATIVE (orange mask, grain, vignette)
          vec3 inv = 1.0 - rawColor.rgb;
          vec3 neg = vec3(0.90, 0.50, 0.22) * (0.22 + 0.78 * pow(inv, vec3(1.15, 1.0, 0.9)));
          neg = mix(neg, vec3(getLuminance(neg)), 0.12);
          neg += (hash21(v_screenUv * u_resolution + fract(u_time) * 100.0) - 0.5) * 0.09;
          vec2 vc = v_screenUv - 0.5;
          neg *= 1.0 - dot(vc, vc) * 0.9;
          fx = clamp(neg, 0.0, 1.0);
        }

        // feathered edge (no hard aliased cut) + soft light-blue rim
        float sgn = inPrism ? dEdge : -dEdge;
        float m = smoothstep(-0.006, 0.006, sgn);
        float gray = getLuminance(rawColor.rgb);
        vec3 dim = mix(rawColor.rgb, vec3(gray), 0.65) * 0.42;
        vec3 outCol = mix(dim, fx, m);
        if (quadOk) outCol += vec3(0.25, 0.55, 1.0) * exp(-dEdge * dEdge / 0.00002) * 0.35;
        fragColor = vec4(mix(rawColor.rgb, outCol, u_prismAlpha), 1.0);
      }
    `;

    function compileShader(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(s));
        gl.deleteShader(s);
        return null;
      }
      return s;
    }

    const glProgram = gl.createProgram();
    gl.attachShader(glProgram, compileShader(gl.VERTEX_SHADER, VS_SOURCE));
    gl.attachShader(glProgram, compileShader(gl.FRAGMENT_SHADER, FS_SOURCE));
    gl.linkProgram(glProgram);

    const uVideoLoc = gl.getUniformLocation(glProgram, "u_video");
    const uResLoc = gl.getUniformLocation(glProgram, "u_resolution");
    const uMirrorLoc = gl.getUniformLocation(glProgram, "u_mirror");
    const uFilterLoc = gl.getUniformLocation(glProgram, "u_filter");
    const uPrismAlphaLoc = gl.getUniformLocation(glProgram, "u_prismAlpha");
    const uV0Loc = gl.getUniformLocation(glProgram, "u_v0");
    const uV1Loc = gl.getUniformLocation(glProgram, "u_v1");
    const uV2Loc = gl.getUniformLocation(glProgram, "u_v2");
    const uV3Loc = gl.getUniformLocation(glProgram, "u_v3");
    const uDensityLoc = gl.getUniformLocation(glProgram, "u_density");
    const uSplatRadiusLoc = gl.getUniformLocation(glProgram, "u_splatRadius");
    const uEdgeGainLoc = gl.getUniformLocation(glProgram, "u_edgeGain");
    const U = n => gl.getUniformLocation(glProgram, n);
    const uZoomL = U("u_zoom"), uTime = U("u_time"), uPointer = U("u_pointer"), uPointerOn = U("u_pointerOn"), uMaskL = U("u_mask"), uMkL = U("u_mk"),
          uMakeupL = U("u_makeup"), uHairAL = U("u_hairA"), uHairBL = U("u_hairB"), uHairCL = U("u_hairC"),
          uHairOnL = U("u_hairOn"), uVibL = U("u_vibrance"), uSmoothL = U("u_smooth");
    function makeTex() {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
      return t;
    }
    const maskTex = makeTex(), mkTex = makeTex();
    const hexRgb = h => { const n = parseInt(h.slice(1), 16); return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; };

    const quadVao = gl.createVertexArray();
    gl.bindVertexArray(quadVao);
    const quadVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const aPosLoc = gl.getAttribLocation(glProgram, "a_position");
    gl.enableVertexAttribArray(aPosLoc);
    gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, 0, 0);

    const videoTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    /* =========================================================================
       4. 2D OVERLAYS: SKELETONS, JUTSU, MAKEUP & PRANA CATCH
       ========================================================================= */
    const vCtx = vecCanvas.getContext("2d");

    // Exact Lip Perimeter Landmark Indices
    const LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61];
    const LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];

    // Hair Palettes (Lustra Blend Presets)
    const HAIR_PALETTES = {
      rosegold: ["#c9808b", "#ff758c"],
      sunset: ["#ff3d2e", "#ff9a3c", "#ffd166"],
      cyberblue: ["#0066ff", "#00f0ff"],
      emerald: ["#169b62", "#2dd4bf"],
      cherry: ["#2b0612", "#8e1538"],
      platinum: ["#d5dae4", "#f1f5f9"]
    };

    function spawnPranaGem() {
      state.gems.push({
        x: Math.random() * (vecCanvas.width - 60) + 30,
        y: -30,
        r: 22,
        vy: 2.2 + Math.random() * 2.0,
        color: ["#38bdf8", "#ec4899", "#a855f7", "#eab308"][Math.floor(Math.random() * 4)]
      });
    }

    
    /* Pinch countdown removed per user request; capture strictly via on-screen buttons and Space/C */

    function render2DOverlays() {
      vCtx.clearRect(0, 0, vecCanvas.width, vecCanvas.height);
      const w = vecCanvas.width;
      const h = vecCanvas.height;

      // 1. SKELETONS: HANDS (Default OFF)
      if (state.showHands && !state.capture && state.handData.length > 0) {
        vCtx.save();
        state.handData.forEach(hand => {
          vCtx.strokeStyle = hand.isPinching ? "rgba(59, 130, 246, 0.95)" : "rgba(148, 163, 184, 0.35)";
          vCtx.lineWidth = hand.isPinching ? 2.5 : 1.5;

          const BONES = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [5, 9], [9, 10], [10, 11], [11, 12],
            [9, 13], [13, 14], [14, 15], [15, 16],
            [13, 17], [17, 18], [18, 19], [19, 20],
            [0, 17]
          ];

          BONES.forEach(([i, j]) => {
            const p1 = hand.landmarks[i];
            const p2 = hand.landmarks[j];
            if (!p1 || !p2) return;
            const s1 = landmarkToScreen(p1);
            const s2 = landmarkToScreen(p2);
            vCtx.beginPath();
            vCtx.moveTo(s1.x, s1.y);
            vCtx.lineTo(s2.x, s2.y);
            vCtx.stroke();
          });

          hand.landmarks.forEach((pt, idx) => {
            const s = landmarkToScreen(pt);
            vCtx.beginPath();
            vCtx.arc(s.x, s.y, (idx === 4 || idx === 8) ? 4.5 : 2.5, 0, Math.PI * 2);
            vCtx.fillStyle = (idx === 4 || idx === 8) ? "#3b82f6" : "#94a3b8";
            vCtx.fill();
          });
        });
        vCtx.restore();
      }

      // 2. SKELETONS: FULL BODY, HEAD & HANDS (Default OFF)
      if (state.showPose && !state.capture && state.poseLandmarks) {
        vCtx.save();
        const P = state.poseLandmarks;
        const vis = i => (P[i] && P[i].visibility != null) ? P[i].visibility : 1;
        const inFrame = i => P[i] && P[i].y >= -0.05 && P[i].y <= 1.05;
        const valid = i => inFrame(i) && vis(i) > 0.40;

        // Head / Face features from Pose
        const HEAD_BONES = [
          [0, 1], [1, 2], [2, 3], [3, 7],         // Left eye to ear
          [0, 4], [4, 5], [5, 6], [6, 8],         // Right eye to ear
          [9, 10], [0, 9], [0, 10]                 // Mouth & nose-mouth
        ];
        // Torso, Arms & Hands
        const UPPER_BONES = [
          [11, 12], [11, 13], [13, 15],           // Left arm
          [12, 14], [14, 16],                     // Right arm
          [15, 17], [15, 19], [15, 21], [17, 19], // Left wrist / fingers
          [16, 18], [16, 20], [16, 22], [18, 20]  // Right wrist / fingers
        ];
        // Lower body (only drawn if hips are actually visible)
        const hipsVisible = valid(23) && valid(24) && P[23].y < 0.92 && P[24].y < 0.92;
        const LOWER_BONES = hipsVisible ? [
          [11, 23], [12, 24], [23, 24],
          [23, 25], [24, 26], [25, 27], [26, 28],
          [27, 29], [29, 31], [27, 31],
          [28, 30], [30, 32], [28, 32]
        ] : [];

        // Draw neck line if shoulders and nose/face are visible
        if (valid(0) && valid(11) && valid(12)) {
          const sNose = landmarkToScreen(P[0]);
          const sMidSh = {
            x: (landmarkToScreen(P[11]).x + landmarkToScreen(P[12]).x) * 0.5,
            y: (landmarkToScreen(P[11]).y + landmarkToScreen(P[12]).y) * 0.5
          };
          vCtx.strokeStyle = "rgba(56, 189, 248, 0.50)";
          vCtx.lineWidth = 2.0;
          vCtx.beginPath();
          vCtx.moveTo(sNose.x, sNose.y);
          vCtx.lineTo(sMidSh.x, sMidSh.y);
          vCtx.stroke();
        }

        // Draw bones
        const drawBones = (bones, strokeStyle, width) => {
          vCtx.strokeStyle = strokeStyle;
          vCtx.lineWidth = width;
          bones.forEach(([i, j]) => {
            if (!valid(i) || !valid(j)) return;
            const s1 = landmarkToScreen(P[i]);
            const s2 = landmarkToScreen(P[j]);
            vCtx.beginPath();
            vCtx.moveTo(s1.x, s1.y);
            vCtx.lineTo(s2.x, s2.y);
            vCtx.stroke();
          });
        };

        drawBones(HEAD_BONES, "rgba(245, 158, 11, 0.65)", 1.6);
        drawBones(UPPER_BONES, "rgba(56, 189, 248, 0.65)", 2.0);
        if (LOWER_BONES.length > 0) drawBones(LOWER_BONES, "rgba(99, 102, 241, 0.55)", 2.0);

        // Draw joint nodes
        const drawJoints = (indices, fillStyle, radius) => {
          vCtx.fillStyle = fillStyle;
          indices.forEach(idx => {
            if (!valid(idx)) return;
            const s = landmarkToScreen(P[idx]);
            vCtx.beginPath();
            vCtx.arc(s.x, s.y, radius, 0, Math.PI * 2);
            vCtx.fill();
          });
        };

        drawJoints([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "#f59e0b", 2.5); // Face/head (gold)
        drawJoints([11, 12, 13, 14, 15, 16], "#38bdf8", 3.5);           // Shoulders, elbows, wrists (cyan)
        drawJoints([17, 18, 19, 20, 21, 22], "#67e8f9", 2.2);           // Hand fingers (light cyan)
        if (hipsVisible) {
          drawJoints([23, 24, 25, 26, 27, 28, 29, 30, 31, 32], "#818cf8", 3.0); // Hips/legs (indigo)
        }

        // If face landmarks are available, draw refined lip contour
        if (state.faceLandmarks) {
          const FL = state.faceLandmarks;
          vCtx.strokeStyle = "rgba(244, 63, 94, 0.60)";
          vCtx.lineWidth = 1.2;
          vCtx.beginPath();
          LIPS_OUTER.forEach((idx, i) => {
            if (!FL[idx]) return;
            const pt = landmarkToScreen(FL[idx]);
            i === 0 ? vCtx.moveTo(pt.x, pt.y) : vCtx.lineTo(pt.x, pt.y);
          });
          vCtx.stroke();
        }

        vCtx.restore();
      }

      // 3. SCENE 1: SPATIAL PRISM QUAD RETICLE (Clean Outline & Precision Corner Nodes)
      if (state.prismOpacity > 0.01 && state.scene === "PRISM") {
        vCtx.save();
        vCtx.globalAlpha = state.prismOpacity;

        const pTL = { x: state.corners.tl.x * w, y: state.corners.tl.y * h };
        const pTR = { x: state.corners.tr.x * w, y: state.corners.tr.y * h };
        const pBR = { x: state.corners.br.x * w, y: state.corners.br.y * h };
        const pBL = { x: state.corners.bl.x * w, y: state.corners.bl.y * h };

        // Color coding: emerald on shutter armed, amber gold on pinch lock, platinum pearl on active tracking
        const mainColor = state.shutterArmed ? "rgba(16, 185, 129, 0.95)" : state.prismLocked ? "rgba(245, 158, 11, 0.95)" : "rgba(248, 250, 252, 0.88)";
        const haloColor = state.shutterArmed ? "rgba(16, 185, 129, 0.28)" : state.prismLocked ? "rgba(245, 158, 11, 0.28)" : "rgba(248, 250, 252, 0.22)";

        // Continuous clean quad outline
        vCtx.strokeStyle = mainColor;
        vCtx.lineWidth = 1.8;
        vCtx.beginPath();
        vCtx.moveTo(pTL.x, pTL.y);
        vCtx.lineTo(pTR.x, pTR.y);
        vCtx.lineTo(pBR.x, pBR.y);
        vCtx.lineTo(pBL.x, pBL.y);
        vCtx.closePath();
        vCtx.stroke();

        // Precision corner dots (outer soft glow + solid inner node)
        [pTL, pTR, pBR, pBL].forEach(pt => {
          // Soft luminous outer halo
          vCtx.fillStyle = haloColor;
          vCtx.beginPath();
          vCtx.arc(pt.x, pt.y, 6.0, 0, Math.PI * 2);
          vCtx.fill();

          // Crisp solid center node
          vCtx.fillStyle = mainColor;
          vCtx.beginPath();
          vCtx.arc(pt.x, pt.y, 3.2, 0, Math.PI * 2);
          vCtx.fill();
        });

        vCtx.restore();
      }

      if (state.scene === "MAKEUP") drawMakeup(w, h);
      if (state.scene === "MYSTIC") drawMystic(w, h);
      if (state.scene === "PRISM" && state.filter === "PRISM_PLEXUS" && state.prismOpacity > 0.01) {
        vCtx.save();
        const c2d = state.corners2D;
        if (c2d) {
          vCtx.beginPath();
          vCtx.moveTo(c2d.tl.x, c2d.tl.y);
          vCtx.lineTo(c2d.tr.x, c2d.tr.y);
          vCtx.lineTo(c2d.br.x, c2d.br.y);
          vCtx.lineTo(c2d.bl.x, c2d.bl.y);
          vCtx.closePath();
          vCtx.clip();
        }
        drawProximityPOP(vCtx, w, h, performance.now() * 0.001);
        vCtx.restore();
      }
      if (state.scene === "GAME") drawGame(w, h);
      if (state.scene === "MUSIC") drawMusic(w, h);
      if (state.showFaceMesh && !state.capture && state.faceLandmarks) drawFaceMesh();
            if (state.shotFlash > 0) { vCtx.fillStyle = `rgba(255,255,255,${state.shotFlash * 0.6})`; vCtx.fillRect(0, 0, w, h); state.shotFlash -= 0.08; }
    }

    /* ---------- shared helpers ---------- */
    const TAU = Math.PI * 2;
    const maskCanvas = document.createElement("canvas"), maskCtx = maskCanvas.getContext("2d");
    const mkCanvas = document.createElement("canvas"), mkCtx = mkCanvas.getContext("2d");
    let prevHair = null, prevSkin = null, prevPerson = null;
    Object.assign(state, { trails: {}, lives: 3, streak: 0, level: 1, over: false, flash: 0, rings: [], headTrail: [],
      maskDirty: false, capture: false, lastHandsTime: -1e9, prevActive: false, shutterArmed: false, lastSnapshotBlob: null, lastSnapshotUrl: null, pinchSince: 0, lastShot: -1e9, shotFlash: 0, zoom: 2.5, skinSmooth: 0.6, best: 0, gmode: "ARCADE", fbGood: true,
      lipOpacity: 0.35,
      blushColor: "none",
      blushOpacity: 0.65, plexusRadius: 65, flockCount: 1200, instrument: "PIANO", octaveOffset: 0, pianoLevel: "FLOATING", flockSpeed: 1.0, gameGalaxy: false });
    try { state.best = +localStorage.getItem("divya_best") || 0; } catch (e) {}
    const badge = t => { const b = document.getElementById("status-badge"); if (b.textContent !== t) b.textContent = t; };
    function glow(c, x, y, r, a, b) {
      const g = c.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, a); g.addColorStop(1, b);
      c.fillStyle = g; c.beginPath(); c.arc(x, y, r, 0, TAU); c.fill();
    }
    function star(c, n, r, k) {
      c.beginPath();
      for (let i = 0; i <= n; i++) { const a = i * k * TAU / n; i ? c.lineTo(Math.cos(a) * r, Math.sin(a) * r) : c.moveTo(Math.cos(a) * r, Math.sin(a) * r); }
      c.stroke();
    }
    function bolt(x1, y1, x2, y2, d, depth) {
      const pts = [[x1, y1], [x2, y2]];
      for (let i = 0; i < depth; i++) {
        for (let j = pts.length - 2; j >= 0; j--) {
          const a = pts[j], b = pts[j + 1];
          pts.splice(j + 1, 0, [(a[0] + b[0]) / 2 + (Math.random() - 0.5) * d, (a[1] + b[1]) / 2 + (Math.random() - 0.5) * d]);
        }
        d *= 0.5;
      }
      return pts;
    }
    function strokePts(c, pts) { c.beginPath(); pts.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.stroke(); }
    function emit(x, y, vx, vy, color, size) { if (state.portalSparks.length < 400) state.portalSparks.push({ x, y, vx, vy, color, size: size || 3, life: 1 }); }

    function drawFaceMesh() {
      const L = state.faceLandmarks, conns = FaceLandmarker.FACE_LANDMARKS_TESSELATION || [];
      vCtx.save(); vCtx.strokeStyle = "rgba(56,189,248,0.35)"; vCtx.lineWidth = 0.6; vCtx.beginPath();
      conns.forEach(({ start, end }) => {
        const a = landmarkToScreen(L[start]), b = landmarkToScreen(L[end]);
        vCtx.moveTo(a.x, a.y); vCtx.lineTo(b.x, b.y);
      });
      vCtx.stroke(); vCtx.restore();
    }

    /* ---------- HAIR & FACE: lips / blush / bindi go to a mask the shader blends ---------- */
    function drawMakeup(w, h) {
      const hw = w >> 1, hh = h >> 1;
      if (mkCanvas.width !== hw) { mkCanvas.width = hw; mkCanvas.height = hh; }
      mkCtx.setTransform(1, 0, 0, 1, 0, 0);
      mkCtx.clearRect(0, 0, hw, hh);
      if (!state.faceLandmarks) return;
      mkCtx.setTransform(0.5, 0, 0, 0.5, 0, 0);
      const L = state.faceLandmarks, P = i => landmarkToScreen(L[i]), c = mkCtx;
      const fw = Math.hypot(P(234).x - P(454).x, P(234).y - P(454).y);
      if (state.lipColor !== "none") {
        c.save(); c.shadowColor = state.lipColor; c.shadowBlur = fw * 0.02; c.fillStyle = state.lipColor; c.globalAlpha = (state.lipOpacity != null ? state.lipOpacity : 0.35);
        c.beginPath();
        LIPS_OUTER.forEach((idx, i) => { const p = P(idx); i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y); });
        LIPS_INNER.slice().reverse().forEach(idx => { const p = P(idx); c.lineTo(p.x, p.y); });
        c.closePath(); c.fill(); c.restore();
        const g = P(17), mw = Math.hypot(P(61).x - P(291).x, P(61).y - P(291).y);
        c.save(); c.globalAlpha = 0.35; glow(c, g.x, g.y - mw * 0.03, mw * 0.2, "rgba(255,255,255,0.6)", "rgba(255,255,255,0)"); c.restore();
      }
      if (state.blushColor && state.blushColor !== "none") {
        const hexToRgb = hex => {
          let h = hex.replace("#", "");
          if (h.length === 3) h = h.split("").map(x => x + x).join("");
          const num = parseInt(h, 16);
          return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
        };
        const [br, bg, bb] = hexToRgb(state.blushColor);
        const blushAlpha = (state.blushOpacity != null ? state.blushOpacity : 0.65) * 0.55;
        [205, 425].forEach(i => {
          const p = P(i);
          c.save();
          c.globalAlpha = 1.0;
          glow(c, p.x, p.y, fw * 0.18, `rgba(${br},${bg},${bb},${blushAlpha})`, `rgba(${br},${bg},${bb},0)`);
          c.restore();
        });
      }
      const f = P(9); c.save(); c.globalAlpha = 0.9; c.fillStyle = "#f59e0b";
      c.beginPath(); c.arc(f.x, f.y, fw * 0.022, 0, TAU); c.fill(); c.restore();
    }

    /* ---------- MYSTIC: additive glow, branching lightning, particle vortex ---------- */
    function shield(c, p, span, t) {
      const r = span * 1.7;
      glow(c, p.x, p.y, r * 1.15, "rgba(255,150,30,0.28)", "rgba(255,120,0,0)");
      c.save(); c.translate(p.x, p.y);
      c.shadowColor = "#ff9a1f"; c.shadowBlur = 14; c.strokeStyle = "rgba(255,176,60,0.95)"; c.lineWidth = 2.4;
      c.beginPath(); c.arc(0, 0, r, 0, TAU); c.stroke();
      c.lineWidth = 1.3; c.beginPath(); c.arc(0, 0, r * 0.86, 0, TAU); c.stroke();
      c.setLineDash([3, 9]); c.rotate(t); c.beginPath(); c.arc(0, 0, r * 0.93, 0, TAU); c.stroke(); c.setLineDash([]); c.rotate(-t);
      for (let a = 0; a < 48; a++) {
        const ang = a * TAU / 48 - t * 0.7, l = a % 4 === 0 ? r * 0.12 : r * 0.05;
        c.beginPath(); c.moveTo(Math.cos(ang) * r * 0.86, Math.sin(ang) * r * 0.86);
        c.lineTo(Math.cos(ang) * (r * 0.86 - l), Math.sin(ang) * (r * 0.86 - l)); c.stroke();
      }
      c.strokeStyle = "#ffe9a8"; c.lineWidth = 1.6;
      c.rotate(t * 0.5); star(c, 8, r * 0.7, 3); c.rotate(-t * 1.1); star(c, 5, r * 0.5, 2); c.rotate(t * 1.6); star(c, 12, r * 0.3, 5);
      c.restore();
      glow(c, p.x, p.y, r * 0.16, "rgba(255,255,255,0.95)", "rgba(255,200,80,0)");
      if (Math.random() < 0.7) {
        const a = Math.random() * TAU;
        emit(p.x + Math.cos(a) * r, p.y + Math.sin(a) * r, Math.cos(a) * 2 + (Math.random() - .5) * 2, Math.sin(a) * 2 - 1, "#ffb347", 2.5);
      }
    }
    function chidori(c, p, span, tips) {
      glow(c, p.x, p.y, span * 1.6, "rgba(120,190,255,0.55)", "rgba(40,100,255,0)");
      c.lineCap = "round"; c.lineJoin = "round";
      const arcs = tips.map(tp => [p, tp, span * 0.8]);
      for (let i = 0; i < 4; i++) {
        const a = Math.random() * TAU, d = span * (1.2 + Math.random() * 1.2);
        arcs.push([p, { x: p.x + Math.cos(a) * d, y: p.y + Math.sin(a) * d }, span * 0.9]);
      }
      for (let i = 0; i < tips.length - 1; i++) if (Math.random() < 0.6) arcs.push([tips[i], tips[i + 1], span * 0.35]);
      arcs.forEach(([a, b, d]) => {
        const pts = bolt(a.x, a.y, b.x, b.y, d, 5);
        [[8, "rgba(60,120,255,0.28)"], [3, "rgba(140,210,255,0.8)"], [1.3, "#fff"]].forEach(([lw, col]) => { c.lineWidth = lw; c.strokeStyle = col; strokePts(c, pts); });
      });
      if (Math.random() < 0.8) { const tp = tips[Math.floor(Math.random() * 5)]; emit(tp.x, tp.y, (Math.random() - .5) * 6, (Math.random() - .5) * 6, "#9fdcff", 2.5); }
      glow(c, p.x, p.y, span * 0.4, "rgba(255,255,255,0.9)", "rgba(150,210,255,0)");
    }
    function rasengan(c, p, span, t) {
      const R = span * 1.15, N = 150;
      glow(c, p.x, p.y, R * 1.5, "rgba(80,170,255,0.35)", "rgba(40,100,255,0)");
      for (let i = 0; i < N; i++) {
        const lat = Math.acos(1 - 2 * (i + 0.5) / N), lon = i * 2.399963 + t * (1.2 + (i % 3) * 0.3);
        const x = R * Math.sin(lat) * Math.cos(lon), z3 = R * Math.sin(lat) * Math.sin(lon), y3 = R * Math.cos(lat);
        const y = y3 * 0.88 - z3 * 0.48, z = y3 * 0.48 + z3 * 0.88, d = (z / R + 1) / 2;
        c.fillStyle = `rgba(${150 + d * 105},${210 + d * 45},255,${0.25 + d * 0.65})`;
        c.beginPath(); c.arc(p.x + x, p.y + y, 1 + d * 2.4, 0, TAU); c.fill();
      }
      c.lineCap = "round";
      for (let k = 0; k < 3; k++) {
        c.strokeStyle = "rgba(170,225,255,0.7)"; c.lineWidth = 2; c.beginPath();
        for (let u = 0; u <= 1; u += 0.025) {
          const a = k * TAU / 3 + u * 9 - t * 6, rr = R * (1 - u * 0.9);
          const x = p.x + Math.cos(a) * rr, y = p.y + Math.sin(a) * rr * 0.55;
          u ? c.lineTo(x, y) : c.moveTo(x, y);
        }
        c.stroke();
      }
      glow(c, p.x, p.y, R * 0.4, "rgba(255,255,255,0.95)", "rgba(120,200,255,0)");
    }
    function portal(c, p, span, t) {
      const r = span * 1.9;
      c.globalCompositeOperation = "source-over";
      glow(c, p.x, p.y, r, "rgba(10,6,24,0.6)", "rgba(10,6,24,0.25)");
      c.globalCompositeOperation = "lighter";
      c.save(); c.translate(p.x, p.y); c.shadowColor = "#ea580c"; c.shadowBlur = 18;
      [[1, 3.5, "#ffb347"], [0.94, 2, "#ffe08a"]].forEach(([s, lw, col], i) => {
        c.strokeStyle = col; c.lineWidth = lw; c.setLineDash(i ? [10, 14] : []); c.lineDashOffset = -t * 60 * (i ? -1 : 1);
        c.beginPath(); c.arc(0, 0, r * s, 0, TAU); c.stroke();
      });
      c.setLineDash([]); c.shadowBlur = 0; c.lineWidth = 1.6; c.strokeStyle = "rgba(255,190,90,0.55)";
      for (let k = 0; k < 4; k++) {
        c.beginPath();
        for (let u = 0; u <= 1; u += 0.04) { const a = k * TAU / 4 + u * 5 + t * 2.5, rr = r * 0.9 * (1 - u * 0.85); u ? c.lineTo(Math.cos(a) * rr, Math.sin(a) * rr) : c.moveTo(Math.cos(a) * rr, Math.sin(a) * rr); }
        c.stroke();
      }
      c.restore();
      for (let i = 0; i < 2; i++) {
        const a = Math.random() * TAU, cols = ["#ffb347", "#ff7a18", "#ffe08a"];
        emit(p.x + Math.cos(a) * r, p.y + Math.sin(a) * r, -Math.sin(a) * 2.5 + (Math.random() - .5) * 2, Math.cos(a) * 2.5 - 1.5, cols[i + (Math.random() * 2 | 0)], 2.8);
      }
    }
    function drawFlow(c, t) {
      const live = {};
      state.handData.forEach((hd, hi) => [4, 8, 12, 16, 20].forEach(ti => {
        const k = hi + "_" + ti, p = landmarkToScreen(hd.landmarks[ti]);
        (state.trails[k] = state.trails[k] || []).push({ x: p.x, y: p.y }); live[k] = 1;
      }));
      c.lineCap = "round";
      for (const k in state.trails) {
        const tr = state.trails[k];
        if (!live[k] || tr.length > 34) tr.shift();
        if (!tr.length) { delete state.trails[k]; continue; }
        const off = +k.split("_")[1] * 18;
        for (let i = 1; i < tr.length; i++) {
          const u = i / tr.length;
          c.strokeStyle = `hsla(${(t * 90 + i * 7 + off) % 360},100%,${55 + u * 20}%,${u * 0.85})`;
          c.lineWidth = 1 + u * 9;
          c.beginPath(); c.moveTo(tr[i - 1].x, tr[i - 1].y); c.lineTo(tr[i].x, tr[i].y); c.stroke();
        }
        if (live[k]) { const e = tr[tr.length - 1]; glow(c, e.x, e.y, 14, `hsla(${(t * 90 + off) % 360},100%,70%,0.8)`, "rgba(255,255,255,0)"); }
      }

      // Two-hand harmonic energy ribbons connecting both hands
      if (state.handData.length >= 2) {
        const h1 = state.handData[0].landmarks;
        const h2 = state.handData[1].landmarks;
        [4, 8, 12, 16, 20].forEach((ti, idx) => {
          const p1 = landmarkToScreen(h1[ti]);
          const p2 = landmarkToScreen(h2[ti]);
          const d = Math.hypot(p2.x - p1.x, p2.y - p1.y);
          if (d < 380) {
            const alpha = Math.max(0, (1 - d / 380) * 0.72);
            c.save();
            c.strokeStyle = `hsla(${(t * 110 + idx * 45) % 360}, 100%, 72%, ${alpha})`;
            c.lineWidth = 2.4;
            c.beginPath();
            const steps = 16;
            for (let i = 0; i <= steps; i++) {
              const u = i / steps;
              const midWave = Math.sin(u * Math.PI * 3 + t * 4 + idx) * (22 * (1 - u) * u * 4);
              const x = p1.x + (p2.x - p1.x) * u;
              const y = p1.y + (p2.y - p1.y) * u + midWave;
              i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
            }
            c.stroke();
            c.restore();
          }
        });
      }
    }
    /* ---------- MYSTIC 2.0: hand-scaled spells + Alchemy / 3D Point Cloud / Finger Text ---------- */
    // Per-hand measurements: palm, fingertips, openness (fist 0 -> open 1), roll angle, pinch ratio. span scales every spell to the hand.
    function handInfo(hd) {
      const L = hd.landmarks, palm = landmarkToScreen(L[9]), wr = landmarkToScreen(L[0]);
      const raw = Math.hypot(palm.x - wr.x, palm.y - wr.y) || 60;
      const tips = [4, 8, 12, 16, 20].map(i => landmarkToScreen(L[i]));
      const open = Math.max(0, Math.min(1, tips.slice(1).reduce((a, q) => a + Math.hypot(q.x - wr.x, q.y - wr.y), 0) / 4 / raw - 1));
      return { L, palm, wr, raw, tips, open, ang: Math.atan2(palm.y - wr.y, palm.x - wr.x),
        pinch: Math.hypot(tips[0].x - tips[1].x, tips[0].y - tips[1].y) / raw, span: raw * 0.5 * (0.6 + 0.8 * open) };
    }
    function fingersUp(L) {
      const d = (a, b) => Math.hypot(L[a].x - L[b].x, L[a].y - L[b].y); let n = 0;
      [[8, 6], [12, 10], [16, 14], [20, 18]].forEach(([t, k]) => { if (d(t, 0) > d(k, 0) * 1.15) n++; });
      return n + (d(4, 17) > d(3, 17) * 1.1 ? 1 : 0);
    }
    const RUNES = ["☉", "☽", "☿", "♀", "♂", "♃", "♄", "♈", "♉", "♊", "♋", "♌"];

    function drawAlchemicalSigil(c, idx, s) {
      c.save();
      c.strokeStyle = "#ffe9a8";
      c.fillStyle = "#ffe9a8";
      c.lineWidth = Math.max(1.2, s * 0.16);
      c.lineCap = "round";
      c.lineJoin = "round";

      const s2 = s * 0.5;
      const s4 = s * 0.25;

      c.beginPath();
      switch (idx % 12) {
        case 0: // Sun: Outer ring + center dot
          c.arc(0, 0, s2, 0, TAU);
          c.stroke();
          c.beginPath();
          c.arc(0, 0, Math.max(1.5, s * 0.14), 0, TAU);
          c.fill();
          break;
        case 1: // Moon: Crescent
          c.arc(0, 0, s2, -Math.PI * 0.5, Math.PI * 0.5);
          c.arc(s4 * 0.9, 0, s2 * 0.72, Math.PI * 0.5, -Math.PI * 0.5, true);
          c.stroke();
          break;
        case 2: // Mercury: Circle + bottom cross + top crescent
          c.arc(0, 0, s4 * 1.1, 0, TAU);
          c.stroke();
          c.beginPath();
          c.moveTo(0, s4 * 1.1); c.lineTo(0, s2 * 1.15);
          c.moveTo(-s4 * 0.7, s4 * 1.7); c.lineTo(s4 * 0.7, s4 * 1.7);
          c.moveTo(-s4 * 1.1, -s4 * 1.8);
          c.quadraticCurveTo(0, -s4 * 0.9, s4 * 1.1, -s4 * 1.8);
          c.stroke();
          break;
        case 3: // Venus: Circle + bottom cross
          c.arc(0, -s4 * 0.4, s4 * 1.2, 0, TAU);
          c.stroke();
          c.beginPath();
          c.moveTo(0, s4 * 0.8); c.lineTo(0, s2 * 1.25);
          c.moveTo(-s4 * 0.7, s4 * 1.6); c.lineTo(s4 * 0.7, s4 * 1.6);
          c.stroke();
          break;
        case 4: // Mars: Circle + diagonal arrow up-right
          c.arc(-s4 * 0.35, s4 * 0.35, s4 * 1.1, 0, TAU);
          c.stroke();
          c.beginPath();
          c.moveTo(s4 * 0.4, -s4 * 0.4); c.lineTo(s2 * 1.15, -s2 * 1.15);
          c.lineTo(s4 * 0.6, -s2 * 1.15);
          c.moveTo(s2 * 1.15, -s2 * 1.15); c.lineTo(s2 * 1.15, -s4 * 0.6);
          c.stroke();
          break;
        case 5: // Jupiter: Cross and curved 4
          c.moveTo(-s4 * 1.2, -s4 * 0.6);
          c.quadraticCurveTo(-s4 * 0.4, -s4 * 1.4, s4 * 0.3, -s4 * 0.6);
          c.lineTo(s4 * 0.3, s2 * 1.1);
          c.moveTo(-s4 * 1.1, s4 * 0.3); c.lineTo(s2 * 0.95, s4 * 0.3);
          c.stroke();
          break;
        case 6: // Saturn: Top cross with sickle foot
          c.moveTo(-s4 * 0.8, -s2 * 0.9); c.lineTo(-s4 * 0.8, s4 * 0.5);
          c.moveTo(-s2 * 0.9, -s4 * 1.0); c.lineTo(s4 * 0.1, -s4 * 1.0);
          c.quadraticCurveTo(s2 * 0.9, s4 * 0.8, -s4 * 0.4, s2 * 1.1);
          c.stroke();
          break;
        case 7: // Fire: Equilateral Triangle pointing UP
          c.moveTo(0, -s2 * 1.15);
          c.lineTo(s2 * 1.05, s2 * 0.85);
          c.lineTo(-s2 * 1.05, s2 * 0.85);
          c.closePath();
          c.stroke();
          break;
        case 8: // Water: Equilateral Triangle pointing DOWN
          c.moveTo(0, s2 * 1.15);
          c.lineTo(s2 * 1.05, -s2 * 0.85);
          c.lineTo(-s2 * 1.05, -s2 * 0.85);
          c.closePath();
          c.stroke();
          break;
        case 9: // Air: Triangle UP with horizontal crossbar
          c.moveTo(0, -s2 * 1.15);
          c.lineTo(s2 * 1.05, s2 * 0.85);
          c.lineTo(-s2 * 1.05, s2 * 0.85);
          c.closePath();
          c.moveTo(-s2 * 0.7, -s4 * 0.1);
          c.lineTo(s2 * 0.7, -s4 * 0.1);
          c.stroke();
          break;
        case 10: // Earth: Triangle DOWN with horizontal crossbar
          c.moveTo(0, s2 * 1.15);
          c.lineTo(s2 * 1.05, -s2 * 0.85);
          c.lineTo(-s2 * 1.05, -s2 * 0.85);
          c.closePath();
          c.moveTo(-s2 * 0.7, s4 * 0.1);
          c.lineTo(s2 * 0.7, s4 * 0.1);
          c.stroke();
          break;
        case 11: // Ouroboros / Infinity Loop
          c.arc(-s4 * 0.95, 0, s4 * 0.85, 0, TAU);
          c.stroke();
          c.beginPath();
          c.arc(s4 * 0.95, 0, s4 * 0.85, 0, TAU);
          c.stroke();
          break;
      }
      c.restore();
    }
    function drawAlchemy(c, t) {
      const H = state.handData.map(handInfo);
      badge(H.length > 1 ? "ALCHEMY · hands apart = bigger circle · close fists to transmute" : "ALCHEMY · pinch wide = bigger circle · two hands = bind a circle");
      if (!H.length) return;
      let p, r, ang;
      if (H.length > 1) { const a = H[0].palm, b = H[1].palm; p = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; r = Math.hypot(a.x - b.x, a.y - b.y) * 0.5; ang = Math.atan2(b.y - a.y, b.x - a.x); }
      else { const q = H[0]; p = q.palm; r = q.raw * Math.min(1.9, 0.6 + q.pinch * 1.1); ang = q.ang; }
      r = Math.max(30, r);
      const energy = H.reduce((a, q) => a + q.open, 0) / H.length;
      state.alch = (state.alch || 0) + (energy - (state.alch || 0)) * 0.15; const e = state.alch;
      if (e > 0.65) state.alchHigh = true;
      if (state.alchHigh && e < 0.3) { state.alchHigh = false; playAudioTone(660, "triangle", 0.3, 0.2); for (let i = 0; i < 80; i++) { const a = Math.random() * TAU, v = 3 + Math.random() * 8; emit(p.x, p.y, Math.cos(a) * v, Math.sin(a) * v, i % 2 ? "#ffd166" : "#fff3b0", 3.5); } }
      glow(c, p.x, p.y, r * 1.25, `rgba(255,170,40,${0.12 + e * 0.2})`, "rgba(255,120,0,0)");
      c.save(); c.translate(p.x, p.y); c.rotate(ang + t * 0.25);
      c.shadowColor = "#ffb020"; c.shadowBlur = 12; c.strokeStyle = "rgba(255,196,80,0.95)";
      [1, 0.82, 0.5].forEach((k, i) => { c.lineWidth = i ? 1.3 : 2.4; c.beginPath(); c.arc(0, 0, r * k, 0, TAU); c.stroke(); });
      // Procedural vector alchemical sigils (crisp Canvas 2D paths, zero missing unicode/white dots)
      for (let i = 0; i < 12; i++) {
        c.save();
        c.rotate(i * TAU / 12);
        c.translate(0, -r * 0.91);
        drawAlchemicalSigil(c, i, Math.max(10, r * 0.11));
        c.restore();
      }
      c.strokeStyle = "#ffe9a8"; c.lineWidth = 1.6;
      c.save(); c.rotate(-t * 0.6); star(c, 3, r * 0.8, 1); c.rotate(Math.PI); star(c, 3, r * 0.8, 1); c.restore();
      c.save(); c.rotate(t * 0.9); star(c, 4, r * 0.5, 1); c.restore();
      c.restore();
      H.forEach(q => q.tips.forEach(tp => { c.strokeStyle = "rgba(255,200,90,0.35)"; c.lineWidth = 1.2; strokePts(c, bolt(tp.x, tp.y, p.x, p.y, r * 0.15, 3)); }));
      if (H.length > 1) {
        // Transmutation binding arc between both palms
        c.strokeStyle = "rgba(255, 220, 110, 0.75)";
        c.lineWidth = 2.2;
        strokePts(c, bolt(H[0].palm.x, H[0].palm.y, H[1].palm.x, H[1].palm.y, r * 0.22, 4));
        glow(c, p.x, p.y, r * 0.35 * (0.8 + e), "rgba(255, 235, 170, 0.9)", "rgba(255, 170, 40, 0)");
      }
      glow(c, p.x, p.y, r * 0.18 * (0.6 + e), "rgba(255,255,255,0.95)", "rgba(255,190,70,0)");
      if (Math.random() < 0.5 + e * 0.5) { const a = Math.random() * TAU; emit(p.x + Math.cos(a) * r, p.y + Math.sin(a) * r, Math.cos(a), -1.5 - Math.random() * 2, "#ffcf66", 2.2); }
    }

    const CLOUD_N = 1600, CLOUD_KINDS = ["sphere", "torus", "helix", "cube", "galaxy", "lorenz", "knot", "wave", "heart", "vortex"], cloudCache = {};
    const fr = x => x - Math.floor(x);
    function cloudShape(kind) {
      if (cloudCache[kind]) return cloudCache[kind];
      const a = [];
      if (kind === "lorenz") {
        let X = 0.1, Y = 0, Z = 0;
        for (let i = 0; i < CLOUD_N * 4 + 300; i++) { const dx = 10 * (Y - X), dy = X * (28 - Z) - Y, dz = X * Y - 8 / 3 * Z; X += dx * 0.006; Y += dy * 0.006; Z += dz * 0.006; if (i > 300 && i % 4 === 0) a.push([X / 22, (Z - 25) / 27, Y / 27]); }
        while (a.length < CLOUD_N) a.push(a[a.length - 1]);
        return (cloudCache[kind] = a.slice(0, CLOUD_N));
      }
      for (let i = 0; i < CLOUD_N; i++) {
        const u = (i + 0.5) / CLOUD_N, r1 = fr(Math.sin(i * 12.9898) * 43758.5453), r2 = fr(Math.sin(i * 78.233) * 12345.6789); let x, y, z;
        if (kind === "sphere") { const la = Math.acos(1 - 2 * u), lo = i * 2.399963; x = Math.sin(la) * Math.cos(lo); y = Math.cos(la); z = Math.sin(la) * Math.sin(lo); }
        else if (kind === "torus") { const lo = i * 2.399963, th = i * 4.742, q = 0.75 * (1 + 0.38 * Math.cos(th)); x = q * Math.cos(lo); z = q * Math.sin(lo); y = 0.29 * Math.sin(th); }
        else if (kind === "helix") { const an = u * 6 * TAU + (i % 2) * Math.PI; x = Math.cos(an) * 0.5; z = Math.sin(an) * 0.5; y = u * 2 - 1; }
        else if (kind === "cube") { const f = i % 6, ax = f >> 1, sg = f & 1 ? 1 : -1, o = [0, 0, 0]; o[ax] = sg; o[(ax + 1) % 3] = r1 * 2 - 1; o[(ax + 2) % 3] = r2 * 2 - 1; [x, y, z] = o.map(k => k * 0.7); }
        else if (kind === "galaxy") { const rad = Math.pow(u, 0.6), an = rad * 5 + (i % 3) * TAU / 3 + (r1 - 0.5) * 0.4; x = rad * Math.cos(an); z = rad * Math.sin(an); y = (r2 - 0.5) * 0.18 * (1 - rad); }
        else if (kind === "knot") { const t = u * TAU, tr = 0.07 * Math.sqrt(r1), ta = r2 * TAU; x = (Math.sin(t) + 2 * Math.sin(2 * t)) / 3.2 + tr * Math.cos(ta); y = (Math.cos(t) - 2 * Math.cos(2 * t)) / 3.2 + tr * Math.sin(ta); z = -Math.sin(3 * t) / 3.2 + tr * Math.cos(ta * 2); }
        else if (kind === "wave") { const g = 40, gx = (i % g) / (g - 1) * 2 - 1, gz = (Math.floor(i / g) % g) / (g - 1) * 2 - 1; x = gx; z = gz; y = 0; }
        else if (kind === "heart") { const th = r1 * TAU, rr = Math.sqrt(r2); x = 16 * Math.pow(Math.sin(th), 3) / 17 * rr; y = (13 * Math.cos(th) - 5 * Math.cos(2 * th) - 2 * Math.cos(3 * th) - Math.cos(4 * th)) / 17 * rr; z = (fr(r1 * 91.7) - 0.5) * 0.5 * (1 - rr * 0.6); }
        else { const rad = 0.12 + u * 0.75, an = u * 22 + i; x = rad * Math.cos(an); z = rad * Math.sin(an); y = u * 2 - 1; }
        a.push([x, y, z]);
      }
      return (cloudCache[kind] = a);
    }
    function drawCloud3D(c, t) {
      const H = state.handData.map(handInfo);
      if (!H.length) { badge("3D CLOUD · fingers up pick the shape (0-10, use both hands) · hands apart = size"); return; }
      let p, R, roll, yaw;
      if (H.length > 1) { const a = H[0].palm, b = H[1].palm; p = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; R = Math.hypot(a.x - b.x, a.y - b.y) * 0.5; roll = Math.atan2(b.y - a.y, b.x - a.x); yaw = t * 0.6 + (a.z - b.z) * 6; }
      else { const q = H[0]; p = q.palm; R = q.raw * (0.8 + q.open * 1.2); roll = q.ang - Math.PI / 2; yaw = t * 0.6 + (q.tips[1].x - q.tips[0].x) / q.raw * 0.8; }
      R = Math.max(30, R);
      const n = Math.min(10, H.reduce((a, q) => a + fingersUp(q.L), 0)), kind = n === 0 ? "sphere" : CLOUD_KINDS[n - 1], k = n === 0 ? 0.35 : 1;
      badge(`3D CLOUD · ${n} FINGER${n === 1 ? "" : "S"} = ${kind.toUpperCase()} · hands apart = size`);
      const tgt = cloudShape(kind), pts = state.cloudPts || (state.cloudPts = Array.from({ length: CLOUD_N }, () => ({ x: 0, y: 0, z: 0, ox: 0, oy: 0 })));
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cr = Math.cos(roll), sr = Math.sin(roll), cx = Math.cos(0.4), sx = Math.sin(0.4);
      const tips = []; H.forEach(q => q.tips.forEach(tp => tips.push(tp)));
      for (let i = 0; i < CLOUD_N; i++) {
        const q = pts[i], g = tgt[i];
        const gy = kind === "wave" ? 0.16 * Math.sin(t * 3 + g[0] * 6) * Math.cos(t * 2 + g[2] * 6) : g[1];
        q.x += (g[0] * k - q.x) * 0.12; q.y += (gy * k - q.y) * 0.12; q.z += (g[2] * k - q.z) * 0.12;
        const X = q.x * cy + q.z * sy; let Z = -q.x * sy + q.z * cy; const Y2 = q.y * cx - Z * sx; Z = q.y * sx + Z * cx;
        const X3 = X * cr - Y2 * sr, Y3 = X * sr + Y2 * cr, f = 1 / (1 - Z * 0.3);
        const px = p.x + X3 * R * f + q.ox, py = p.y + Y3 * R * f + q.oy;
        for (const tp of tips) { const dx = px - tp.x, dy = py - tp.y, d = Math.hypot(dx, dy); if (d < 70 && d > 0.1) { q.ox += dx / d * (70 - d) * 0.12; q.oy += dy / d * (70 - d) * 0.12; } }
        q.ox *= 0.92; q.oy *= 0.92;
        const d = Math.max(0, Math.min(1, (Z + 1) / 2)), sz = 1 + d * 2.2;
        c.fillStyle = `hsla(${(190 + d * 90 + i * 0.05 + t * 20) % 360},100%,${55 + d * 25}%,${0.3 + d * 0.65})`; c.fillRect(px - sz / 2, py - sz / 2, sz, sz);
      }
      glow(c, p.x, p.y, R * 0.3, "rgba(160,200,255,0.25)", "rgba(80,120,255,0)");
    }

    /* TouchDesigner-style feedback loop: the canvas is redrawn into itself each frame, slightly zoomed + rotated + faded */
    const fb = { c: document.createElement("canvas"), prev: {} };
    function drawFeedback(c, w, h, t) {
      badge("LIGHT PAINTER · paint with fingertips · pinch = fat brush · fist = clear");
      if (fb.c.width !== w || fb.c.height !== h) { fb.c.width = w; fb.c.height = h; }
      const f = fb.c.getContext("2d");
      f.save(); f.globalCompositeOperation = "copy"; f.globalAlpha = 0.955; f.translate(w / 2, h / 2); f.rotate(0.004); f.scale(1.012, 1.012); f.drawImage(fb.c, -w / 2, -h / 2); f.restore();
      f.save(); f.globalCompositeOperation = "destination-out"; f.fillStyle = "rgba(0,0,0,0.02)"; f.fillRect(0, 0, w, h); f.restore();
      f.save(); f.globalCompositeOperation = "lighter"; f.lineCap = "round";
      const live = {};
      state.handData.forEach((hd, hi) => {
        const q = handInfo(hd); if (q.open < 0.12) { f.clearRect(0, 0, w, h); return; }
        q.tips.forEach((tp, i) => {
          const k = hi + "_" + i, pv = fb.prev[k]; live[k] = 1;
          if (pv) { f.strokeStyle = `hsla(${(t * 60 + i * 55 + hi * 90) % 360},100%,60%,0.9)`; f.lineWidth = (i === 1 ? 10 : 5) * (q.pinch < 0.45 ? 2.5 : 1); f.beginPath(); f.moveTo(pv.x, pv.y); f.lineTo(tp.x, tp.y); f.stroke(); }
          fb.prev[k] = { x: tp.x, y: tp.y };
        });
      });
      for (const k in fb.prev) if (!live[k]) delete fb.prev[k];
      f.restore();
      c.save(); c.globalCompositeOperation = "lighter"; c.drawImage(fb.c, 0, 0); c.restore();
    }

    /* Bioluminescent Boid Fish Swarm (Cosmic Schooling Particle Simulation):
       2,400 schooling micro-boids with aquatic tail undulation, vortex gravitation, and shockwave scatter */
    const flock = [];
    function drawFlock(c, w, h, t) {
      badge("BIOLUMINESCENT SWARM · schooling fish swarm · pinch = orbital vortex · fist = shockwave scatter");
      const targetCount = Math.min(Math.max(state.flockCount || 1200, 200), 2400);
      while (flock.length < targetCount) {
        flock.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 2,
          vy: (Math.random() - 0.5) * 2,
          z: 0.5 + Math.random() * 0.9,
          phase: Math.random() * TAU,
          id: flock.length
        });
      }

      const attractors = [];
      const repellers = [];
      state.handData.forEach(hd => {
        const q = handInfo(hd);
        if (q.open < 0.22) {
          // Fist / push: scatter shockwave
          repellers.push({ x: q.palm.x, y: q.palm.y, r: Math.max(120, q.raw * 2.8) });
        } else {
          // Pinch = tight gravitational vortex; Open hand = schooling follow
          const isPinch = q.pinch < 0.45;
          attractors.push({
            x: q.tips[1].x,
            y: q.tips[1].y,
            pinch: isPinch,
            k: isPinch ? 3.4 : 1.2,
            vortex: isPinch ? 2.8 : 0.8
          });
        }
      });

      c.save();
      c.globalCompositeOperation = "lighter";
      c.lineCap = "round";

      for (let i = 0; i < targetCount; i++) {
        const b = flock[i];
        const z = b.z;

        // Multi-frequency turbulent flow current (creates organic fluid schooling filaments)
        const a1 = Math.sin(b.x * 0.0035 + t * 0.6) * Math.cos(b.y * 0.0042 - t * 0.45) * TAU;
        const a2 = Math.cos(b.x * 0.008 - t * 0.3) * Math.sin(b.y * 0.0075 + t * 0.5) * Math.PI;
        b.vx += (Math.cos(a1) * 0.22 + Math.cos(a2) * 0.12) * z;
        b.vy += (Math.sin(a1) * 0.22 + Math.sin(a2) * 0.12) * z;

        // Attractor gravitation and orbital vortex
        let closestDist = 1e9;
        let activeAttractor = null;
        for (let j = 0; j < attractors.length; j++) {
          const at = attractors[j];
          const dx = at.x - b.x;
          const dy = at.y - b.y;
          const d = Math.hypot(dx, dy);
          if (d < closestDist) {
            closestDist = d;
            activeAttractor = at;
          }
        }

        if (activeAttractor) {
          const dx = activeAttractor.x - b.x;
          const dy = activeAttractor.y - b.y;
          const dist = closestDist + 8.0;
          const nx = dx / dist;
          const ny = dy / dist;

          // Pull inwards + spin orthogonally (vortex swirl)
          const pullForce = (activeAttractor.pinch ? Math.min(4.5, 380 / dist) : Math.min(2.5, 200 / dist)) * activeAttractor.k * 0.4;
          const swirlForce = activeAttractor.vortex * (activeAttractor.pinch ? 1.6 : 0.8);

          b.vx += (nx * pullForce - ny * swirlForce) * z;
          b.vy += (ny * pullForce + nx * swirlForce) * z;
        }

        // Repeller panic scattering
        for (let j = 0; j < repellers.length; j++) {
          const rep = repellers[j];
          const dx = b.x - rep.x;
          const dy = b.y - rep.y;
          const dist = Math.hypot(dx, dy) + 0.1;
          if (dist < rep.r) {
            const push = (1.0 - dist / rep.r) * 6.0;
            b.vx += (dx / dist) * push;
            b.vy += (dy / dist) * push;
          }
        }

        // Velocity damping and speed limiting
        b.vx *= 0.945;
        b.vy *= 0.945;
        const speed = Math.hypot(b.vx, b.vy);
        const maxSpeed = 12.0 * z * (state.flockSpeed || 1.0);
        if (speed > maxSpeed) {
          b.vx = (b.vx / speed) * maxSpeed;
          b.vy = (b.vy / speed) * maxSpeed;
        }

        // Screen boundary wrapping
        b.x += b.vx;
        b.y += b.vy;
        if (b.x < -20) b.x = w + 20;
        else if (b.x > w + 20) b.x = -20;
        if (b.y < -20) b.y = h + 20;
        else if (b.y > h + 20) b.y = -20;

        // Aquatic tail flick undulation
        const tailAngle = Math.atan2(b.vy, b.vx);
        const wag = Math.sin(t * 15.0 + b.phase) * (1.2 * z);
        const tailX = b.x - Math.cos(tailAngle) * (speed * 2.8 + 3.0) + Math.sin(tailAngle) * wag;
        const tailY = b.y - Math.sin(tailAngle) * (speed * 2.8 + 3.0) - Math.cos(tailAngle) * wag;

        // Speed-dependent luminescence & color grading
        const spRatio = Math.min(1.0, speed / 8.0);
        const hue = 190 + spRatio * 25; // 190 (deep cyan) -> 215 (cobalt / stellar white)
        const light = 50 + spRatio * 42; // Up to 92% luminous white on high-speed sprint
        const alpha = (0.35 + spRatio * 0.55) * Math.min(1.0, z);

        // Motion streak / tail
        c.lineWidth = (1.0 + spRatio * 1.5) * z;
        c.strokeStyle = `hsla(${hue}, 95%, ${light}%, ${alpha})`;
        c.beginPath();
        c.moveTo(tailX, tailY);
        c.lineTo(b.x, b.y);
        c.stroke();

        // Luminous boid head
        if (spRatio > 0.4 || z > 1.0) {
          c.fillStyle = `hsla(${hue}, 100%, ${Math.min(96, light + 10)}%, ${alpha * 0.9})`;
          c.beginPath();
          c.arc(b.x, b.y, 1.2 * z, 0, TAU);
          c.fill();
        }
      }
      c.restore();
    }

    /* Finger Text: title + subtitle + caption sampled into ~14k particles */
    const txt = { lines: ["DIVYA", "DRISHTI", "SPATIAL VISION STUDIO"], key: "", pts: [] };
    const TXT_HUE = [195, 42, 320], TXT_COL = TXT_HUE.map(hu => Array.from({ length: 10 }, (_, b) => `hsla(${hu * (1 - b / 9) + 20 * b / 9},100%,${62 + b * 2}%,0.92)`));
    function buildText(w, h) {
      const L = txt.lines; txt.key = L.join("|") + w + "x" + h;
      const oc = document.createElement("canvas"); oc.width = w; oc.height = h; const x = oc.getContext("2d");
      const isPortrait = h > w;
      const rel = isPortrait ? [0.18, 0.08, 0.045] : [0.36, 0.11, 0.05];
      const sp = [0, 0.35, 0.28];
      const wt = [900, 700, 600];
      const ys = isPortrait
        ? (L.length === 1 ? [0.42] : L.length === 2 ? [0.38, 0.52] : [0.32, 0.44, 0.54])
        : (L.length === 1 ? [0.5] : L.length === 2 ? [0.45, 0.66] : [0.40, 0.62, 0.74]);
      x.fillStyle = "#fff"; x.textBaseline = "middle"; x.textAlign = "left";
      L.forEach((str, li) => {
        let fs = h * rel[li]; const ls = sp[li];
        const wide = f => { x.font = `${wt[li]} ${f}px system-ui, sans-serif`; return [...str].reduce((a, ch) => a + x.measureText(ch).width + f * ls, 0) - f * ls; };
        const mw = wide(fs); if (mw > w * 0.9) fs *= w * 0.9 / mw;
        let cx = w / 2 - wide(fs) / 2;
        for (const ch of str) { x.fillText(ch, cx, h * ys[li]); cx += x.measureText(ch).width + fs * ls; }
      });
      const d = x.getImageData(0, 0, w, h).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 128) n++;
      const st = Math.max(2, Math.ceil(Math.sqrt(n / 14000))), pts = [];
      for (let y = 0; y < h; y += st) for (let xx = 0; xx < w; xx += st) if (d[(y * w + xx) * 4 + 3] > 128) {
        let li = 0, b = 1e9; ys.forEach((v, i) => { const dd = Math.abs(y - v * h); if (dd < b) { b = dd; li = i; } });
        pts.push({ hx: xx, hy: y, x: xx + (Math.random() - .5) * w * 0.3, y: y + (Math.random() - .5) * h * 0.3, vx: 0, vy: 0, li });
      }
      txt.pts = pts; txt.sz = st >= 3 ? 2.5 : 2;
    }
    function drawTextFX(c, w, h) {
      badge("TEXT · push with fingers · pinch to pull · tap TEXT again to edit lines");
      if (txt.key !== txt.lines.join("|") + w + "x" + h) buildText(w, h);
      const tips = [];
      state.handData.forEach(hd => { const q = handInfo(hd); q.tips.forEach((tp, i) => tips.push({ x: tp.x, y: tp.y, pull: q.pinch < 0.45 && i === 1, big: i === 1 })); });
      const R = Math.min(55, Math.max(24, w * 0.065)), R2 = R * R, z = txt.sz;
      for (const p of txt.pts) {
        let fx = (p.hx - p.x) * 0.045, fy = (p.hy - p.y) * 0.045;
        for (const tp of tips) {
          const dx = p.x - tp.x, dy = p.y - tp.y, d2 = dx * dx + dy * dy;
          if (d2 < R2) { const d = Math.sqrt(d2) + 0.001, k = 1 - d / R, m = tp.big ? 9 : 5;
            if (tp.pull) { fx += (-dx / d * 3 - dy / d * 4) * k; fy += (-dy / d * 3 + dx / d * 4) * k; }
            else { fx += dx / d * k * m; fy += dy / d * k * m; } }
        }
        p.vx = (p.vx + fx) * 0.84; p.vy = (p.vy + fy) * 0.84; p.x += p.vx; p.y += p.vy;
        const m = Math.min(9, (Math.hypot(p.x - p.hx, p.y - p.hy) / 14) | 0);
        c.fillStyle = TXT_COL[p.li][m]; c.fillRect(p.x - z / 2, p.y - z / 2, z, z);
      }
    }

    
    /* Proximity POP (TouchDesigner Biometric Plexus):
       Dynamic distance-threshold network across MediaPipe Pose + Hands with dual-tone gradient */
    function drawProximityPOP(c, w, h, t) {
      badge("PROXIMITY POP · TouchDesigner biometric plexus network · move hands together");
      const P = state.poseLandmarks;
      const H = state.handData;
      const pts = [];

      const addPt = (x, y, type) => {
        if (x >= -20 && x <= w + 20 && y >= -20 && y <= h + 20) {
          pts.push({ x, y, type }); // type: 'hand' | 'head' | 'body'
        }
      };

      // 1. Hands landmarks (21 points each hand + interpolated finger segments)
      if (H && H.length) {
        H.forEach(hd => {
          const lms = hd.landmarks;
          if (!lms) return;
          lms.forEach(lm => {
            const pt = landmarkToScreen(lm);
            addPt(pt.x, pt.y, "hand");
          });
          // Mid-finger interpolation for higher mesh density
          const chains = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20]
          ];
          chains.forEach(ch => {
            for (let i = 0; i < ch.length - 1; i++) {
              const a = landmarkToScreen(lms[ch[i]]);
              const b = landmarkToScreen(lms[ch[i + 1]]);
              addPt((a.x + b.x) * 0.5, (a.y + b.y) * 0.5, "hand");
            }
          });
        });
      }

      // 2. Pose landmarks (face + interpolated body contours)
      if (P && P.length) {
        // Head / Face
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].forEach(i => {
          if (P[i] && (P[i].visibility == null || P[i].visibility > 0.35)) {
            const p = landmarkToScreen(P[i]);
            addPt(p.x, p.y, "head");
          }
        });

        // Skeleton limb interpolation
        const limbs = [
          [11, 12, 4], // shoulder bridge (4 subdivisions)
          [11, 13, 3], [13, 15, 3], // left arm
          [12, 14, 3], [14, 16, 3], // right arm
          [11, 23, 4], [12, 24, 4], // torso sides
          [23, 24, 3], // hip bridge
          [23, 25, 4], [25, 27, 4], // left leg
          [24, 26, 4], [26, 28, 4]  // right leg
        ];
        limbs.forEach(([i1, i2, subs]) => {
          if (P[i1] && P[i2] && (P[i1].visibility == null || P[i1].visibility > 0.3) && (P[i2].visibility == null || P[i2].visibility > 0.3)) {
            const a = landmarkToScreen(P[i1]);
            const b = landmarkToScreen(P[i2]);
            for (let s = 0; s <= subs; s++) {
              const u = s / subs;
              const jx = Math.sin(t * 3.2 + u * 12) * 2.0;
              const jy = Math.cos(t * 3.7 + u * 12) * 2.0;
              addPt(a.x + (b.x - a.x) * u + jx, a.y + (b.y - a.y) * u + jy, "body");
            }
          }
        });
      }

      if (pts.length < 2) return;

      const maxDist = state.plexusRadius || Math.max(48, Math.min(w, h) * 0.082);
      const maxDistSq = maxDist * maxDist;

      c.save();
      c.globalCompositeOperation = "lighter";

      // Proximity POP connections
      c.lineWidth = 1.3;
      const n = pts.length;
      for (let i = 0; i < n; i++) {
        const p1 = pts[i];
        for (let j = i + 1; j < n; j++) {
          const p2 = pts[j];
          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < maxDistSq) {
            const d = Math.sqrt(d2);
            const alpha = Math.pow(1 - d / maxDist, 1.35) * 0.88;

            if (p1.type === "hand" || p2.type === "hand") {
              c.strokeStyle = `rgba(244, 63, 178, ${alpha})`; // Magenta/Pink for hands
            } else if (p1.type === "head" || p2.type === "head") {
              c.strokeStyle = `rgba(217, 70, 239, ${alpha})`; // Violet for head
            } else {
              c.strokeStyle = `rgba(224, 242, 254, ${alpha * 0.9})`; // Frost ice-white/cyan for body
            }

            c.beginPath();
            c.moveTo(p1.x, p1.y);
            c.lineTo(p2.x, p2.y);
            c.stroke();
          }
        }
      }

      // Biometric POP nodes (TouchDesigner square pixel POP aesthetic)
      pts.forEach(p => {
        let nodeColor, haloColor;
        if (p.type === "hand") {
          nodeColor = "#f43fb2";
          haloColor = "rgba(244, 63, 178, 0.4)";
        } else if (p.type === "head") {
          nodeColor = "#d946ef";
          haloColor = "rgba(217, 70, 239, 0.4)";
        } else {
          nodeColor = "#e0f2fe";
          haloColor = "rgba(186, 230, 253, 0.35)";
        }

        // Circular halo
        c.fillStyle = haloColor;
        c.beginPath();
        c.arc(p.x, p.y, 4.5, 0, TAU);
        c.fill();

        // High-contrast central square node
        c.fillStyle = nodeColor;
        c.fillRect(p.x - 1.5, p.y - 1.5, 3, 3);
      });

      c.restore();
    }

    
    // Procedural Cosmic Galaxy Backdrop for Mystic Jutsu Mode
    function drawGalaxyBackground(c, w, h, t) {
      const bg = c.createRadialGradient(w * 0.5, h * 0.48, 15, w * 0.5, h * 0.5, Math.max(w, h) * 0.85);
      bg.addColorStop(0, "#090d24");
      bg.addColorStop(0.38, "#050716");
      bg.addColorStop(0.72, "#020308");
      bg.addColorStop(1.0, "#000002");
      c.fillStyle = bg;
      c.fillRect(0, 0, w, h);

      // Swirling cosmic nebulae
      c.save();
      c.globalCompositeOperation = "screen";
      const nebulae = [
        { hue: 225, r: 0.48, sx: 0.18, sy: 0.12, sp: 0.12 },
        { hue: 285, r: 0.42, sx: -0.15, sy: 0.14, sp: -0.09 },
        { hue: 190, r: 0.38, sx: 0.12, sy: -0.16, sp: 0.15 }
      ];
      nebulae.forEach((neb, idx) => {
        const ang = t * neb.sp + idx * 2.1;
        const nx = w * 0.5 + Math.cos(ang) * (w * neb.sx);
        const ny = h * 0.5 + Math.sin(ang) * (h * neb.sy);
        const rad = Math.max(w, h) * neb.r;
        const g = c.createRadialGradient(nx, ny, 0, nx, ny, rad);
        g.addColorStop(0, `hsla(${neb.hue}, 85%, 26%, 0.45)`);
        g.addColorStop(0.45, `hsla(${neb.hue + 25}, 90%, 14%, 0.20)`);
        g.addColorStop(1, "transparent");
        c.fillStyle = g;
        c.fillRect(0, 0, w, h);
      });

      // Twinkling starfield
      if (!state.stars || state.starsW !== w || state.starsH !== h) {
        state.starsW = w;
        state.starsH = h;
        state.stars = [];
        for (let i = 0; i < 220; i++) {
          state.stars.push({
            x: Math.random() * w,
            y: Math.random() * h,
            r: Math.random() < 0.12 ? (1.6 + Math.random() * 0.8) : (0.5 + Math.random() * 0.8),
            twinkle: Math.random() * TAU,
            color: Math.random() < 0.2 ? "#93c5fd" : Math.random() < 0.4 ? "#fbcfe8" : "#ffffff"
          });
        }
      }
      for (let i = 0; i < state.stars.length; i++) {
        const st = state.stars[i];
        const alpha = 0.35 + Math.sin(t * 3.0 + st.twinkle) * 0.35;
        c.globalAlpha = Math.max(0.12, alpha);
        c.fillStyle = st.color;
        c.beginPath();
        c.arc(st.x, st.y, st.r, 0, TAU);
        c.fill();
      }
      c.restore();
    }

    function drawMystic(w, h) {
      const c = vCtx, t = (state.mandalaAngle += 0.035);
      if (state.mysticGalaxy) {
        c.save();
        c.globalCompositeOperation = "source-over";
        drawGalaxyBackground(c, w, h, t);
        c.restore();
      }
      c.save(); c.globalCompositeOperation = "lighter";
      if (state.spell === "FLOW") drawFlow(c, t);
      else if (state.spell === "ALCHEMY") drawAlchemy(c, t);
      else if (state.spell === "CLOUD") drawCloud3D(c, t);
      else if (state.spell === "TEXT") drawTextFX(c, w, h);
      else if (state.spell === "FEEDBACK") drawFeedback(c, w, h, t);
      else if (state.spell === "FLOCK") drawFlock(c, w, h, t);
      else if (state.spell === "PLEXUS") drawProximityPOP(c, w, h, t);
      else {
        state.handData.forEach(hd => {
          const q = handInfo(hd);
          c.save(); c.globalCompositeOperation = "lighter";
          if (state.spell === "SHIELD") shield(c, q.palm, q.span, t + q.ang);
          else if (state.spell === "CHIDORI") chidori(c, q.palm, q.span, q.tips);
          else if (state.spell === "RASENGAN") rasengan(c, q.palm, q.span, t);
          else portal(c, q.palm, q.span, t);
          c.restore();
        });

        // Two-hand dynamic interaction: energy bridge connecting both hands with proximity fusion & Next-Level Cinematic VFX
        if (state.handData.length >= 2) {
          const q1 = handInfo(state.handData[0]);
          const q2 = handInfo(state.handData[1]);
          const p1 = q1.palm, p2 = q2.palm;
          const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
          const midX = (p1.x + p2.x) * 0.5;
          const midY = (p1.y + p2.y) * 0.5;

          // Continuous proximity factor: 0.0 (distant) to 1.0 (touching / clashing)
          const prox = Math.max(0.0, Math.min(1.0, 1.0 - (dist - 65) / 285));
          const isClash = dist < 180; // Proximity threshold for fusion / clash state

          // 1. UNIVERSAL MULTI-FINGERTIP BRAIDED QUANTUM FILAMENTS
          // Links opposing fingertips (Thumb, Index, Middle, Ring, Pinky) with undulating plasma ribbons
          c.save();
          for (let f = 0; f < 5; f++) {
            const t1 = q1.tips[f], t2 = q2.tips[f];
            if (!t1 || !t2) continue;
            const fDist = Math.hypot(t2.x - t1.x, t2.y - t1.y);
            if (fDist < 360) {
              const fAlpha = (1 - fDist / 360) * (0.35 + prox * 0.65);
              const isIndex = f === 1;
              c.strokeStyle = isIndex
                ? `rgba(255, 255, 255, ${fAlpha})`
                : (f === 0 || f === 4 ? `rgba(96, 165, 250, ${fAlpha * 0.75})` : `rgba(192, 132, 252, ${fAlpha * 0.8})`);
              c.lineWidth = isIndex ? (2.2 + prox * 1.5) : (1.2 + prox * 0.8);
              c.shadowColor = isIndex ? "#60a5fa" : "#c084fc";
              c.shadowBlur = 10 + prox * 12;

              c.beginPath();
              const segs = 18;
              for (let s = 0; s <= segs; s++) {
                const u = s / segs;
                const waveAmp = (16 * prox + 6) * Math.sin(u * Math.PI);
                const waveFreq = u * Math.PI * 3 + t * (6 + f) + f * 1.4;
                const wave = Math.sin(waveFreq) * waveAmp;
                const wx = t1.x + (t2.x - t1.x) * u - ((t2.y - t1.y) / fDist) * wave;
                const wy = t1.y + (t2.y - t1.y) * u + ((t2.x - t1.x) / fDist) * wave;
                s === 0 ? c.moveTo(wx, wy) : c.lineTo(wx, wy);
              }
              c.stroke();
            }
          }
          c.restore();

          // 2. DIPOLAR MAGNETIC FORCEFIELD ARCS BETWEEN PALMS
          c.save();
          const arcAlpha = 0.25 + prox * 0.55;
          const arcColor = state.spell === "SHIELD" ? `rgba(251, 191, 36, ${arcAlpha})`
            : state.spell === "CHIDORI" ? `rgba(147, 197, 253, ${arcAlpha})`
            : state.spell === "RASENGAN" ? `rgba(56, 189, 248, ${arcAlpha})`
            : `rgba(249, 115, 22, ${arcAlpha})`;
          c.strokeStyle = arcColor;
          c.lineWidth = 1.4 + prox * 1.6;
          c.shadowColor = arcColor;
          c.shadowBlur = 12;

          for (const sign of [-1, 1]) {
            c.beginPath();
            c.moveTo(p1.x, p1.y);
            const bow = sign * Math.min(dist * 0.45, 95) * (0.35 + prox * 0.65);
            const nx = -((p2.y - p1.y) / dist) * bow;
            const ny = ((p2.x - p1.x) / dist) * bow;
            c.quadraticCurveTo(midX + nx, midY + ny, p2.x, p2.y);
            c.stroke();
          }
          c.restore();

          // 3. MIDPOINT EXPANDING HARMONIC SHOCKWAVES
          if (prox > 0.2) {
            c.save();
            c.strokeStyle = arcColor;
            c.lineWidth = 1.5;
            for (let ring = 0; ring < 3; ring++) {
              const ringR = ((t * 45 + ring * 25) % (80 + prox * 80));
              const ringAlpha = Math.max(0, 1 - ringR / (80 + prox * 80)) * prox * 0.6;
              c.globalAlpha = ringAlpha;
              c.beginPath();
              c.arc(midX, midY, ringR, 0, TAU);
              c.stroke();
            }
            c.restore();
          }

          // 4. SPELL-SPECIFIC CINEMATIC CLASH / FUSION
          if (state.spell === "CHIDORI") {
            c.save();
            if (isClash) {
              // CHIDORI CLASH: Blinding plasma explosion sphere with branching discharge & multi-layer fractal lightning
              const blastR = 65 + prox * 65;
              glow(c, midX, midY, blastR * 1.4, "rgba(255, 255, 255, 0.98)", "rgba(59, 130, 246, 0)");
              glow(c, midX, midY, blastR * 0.5, "rgba(255, 255, 255, 1.0)", "rgba(147, 197, 253, 0)");

              // Core lightning connection
              c.strokeStyle = "#ffffff";
              c.lineWidth = 4.2;
              c.shadowColor = "#38bdf8";
              c.shadowBlur = 28;
              strokePts(c, bolt(p1.x, p1.y, p2.x, p2.y, 48, 5));

              // 8-directional branching discharge
              for (let i = 0; i < 8; i++) {
                const a = (i * Math.PI) / 4 + t * 4.5;
                const len = blastR + 25 + Math.sin(t * 12 + i) * 20;
                const bx = midX + Math.cos(a) * len;
                const by = midY + Math.sin(a) * len;
                c.strokeStyle = (i % 2 === 0) ? "rgba(255, 255, 255, 0.95)" : "rgba(147, 197, 253, 0.9)";
                c.lineWidth = (i % 2 === 0) ? 3.0 : 1.8;
                strokePts(c, bolt(midX, midY, bx, by, 32, 4));
              }

              // High-velocity orbital electric embers
              for (let i = 0; i < 4; i++) {
                const ea = Math.random() * TAU;
                const er = blastR * (0.4 + Math.random() * 0.6);
                emit(midX + Math.cos(ea) * er, midY + Math.sin(ea) * er, (Math.random() - 0.5) * 9, (Math.random() - 0.5) * 9, "#ffffff", 4);
              }
            } else {
              // High-voltage lightning bridge connecting palms and index fingertips
              c.strokeStyle = "rgba(147, 197, 253, 0.95)";
              c.lineWidth = 3.2;
              c.shadowColor = "#60a5fa";
              c.shadowBlur = 18;
              strokePts(c, bolt(p1.x, p1.y, p2.x, p2.y, Math.min(dist * 0.35, 60), 4));
              if (q1.tips[1] && q2.tips[1]) {
                c.strokeStyle = "rgba(255, 255, 255, 0.92)";
                c.lineWidth = 2.0;
                strokePts(c, bolt(q1.tips[1].x, q1.tips[1].y, q2.tips[1].x, q2.tips[1].y, Math.min(dist * 0.25, 40), 3));
              }
              if (Math.random() < 0.45) {
                emit(midX + (Math.random() - 0.5) * 20, midY + (Math.random() - 0.5) * 20, (Math.random() - 0.5) * 4, (Math.random() - 0.5) * 4, "#93c5fd", 3);
              }
            }
            c.restore();
          } else if (state.spell === "SHIELD") {
            c.save();
            if (isClash) {
              // AEGIS BULWARK FUSION: Dual counter-rotating 12-point Merkaba stars with celestial light beams
              const shieldR = Math.max(q1.span, q2.span) * 2.3;
              glow(c, midX, midY, shieldR * 1.5, "rgba(251, 191, 36, 0.55)", "rgba(245, 158, 11, 0)");
              glow(c, midX, midY, shieldR * 0.35, "rgba(255, 255, 255, 0.95)", "rgba(251, 191, 36, 0)");

              c.translate(midX, midY);
              c.shadowColor = "#f59e0b";
              c.shadowBlur = 24;

              // Outer boundary ring with sacred geometry divisions
              c.strokeStyle = "rgba(254, 240, 138, 0.98)";
              c.lineWidth = 3.5;
              c.beginPath(); c.arc(0, 0, shieldR, 0, TAU); c.stroke();

              // Celestial God-Rays radiating from center
              c.lineWidth = 1.6;
              c.strokeStyle = "rgba(254, 240, 138, 0.4)";
              for (let r = 0; r < 8; r++) {
                const ra = (r * TAU) / 8 + t * 0.6;
                c.beginPath();
                c.moveTo(0, 0);
                c.lineTo(Math.cos(ra) * (shieldR * 1.3), Math.sin(ra) * (shieldR * 1.3));
                c.stroke();
              }

              // Counter-rotating 12-point and 8-point sacred stars
              c.lineWidth = 2.0;
              c.strokeStyle = "#fde047";
              c.save(); c.rotate(t * 1.4); star(c, 12, shieldR * 0.88, 5); c.restore();
              c.save(); c.rotate(-t * 1.8); star(c, 8, shieldR * 0.58, 3); c.restore();
              c.translate(-midX, -midY);

              // Golden ember flare emissions
              for (let i = 0; i < 3; i++) {
                const a = Math.random() * TAU;
                emit(midX + Math.cos(a) * shieldR, midY + Math.sin(a) * shieldR, Math.cos(a) * 3.5, Math.sin(a) * 3.5, "#fde047", 3.2);
              }
            } else {
              // Two-hand harmonic energy barrier bridging between shields
              c.strokeStyle = "rgba(251, 191, 36, 0.88)";
              c.lineWidth = 2.6;
              c.shadowColor = "#f59e0b";
              c.shadowBlur = 16;
              c.beginPath();
              c.moveTo(p1.x, p1.y);
              const cpY = midY + Math.sin(t * 3.5) * 28;
              c.quadraticCurveTo(midX, cpY, p2.x, p2.y);
              c.stroke();
              c.beginPath();
              c.arc(midX, midY, Math.min(dist * 0.24, 52), 0, TAU);
              c.strokeStyle = "rgba(245, 158, 11, 0.7)";
              c.stroke();
            }
            c.restore();
          } else if (state.spell === "RASENGAN") {
            c.save();
            if (isClash) {
              // OODAMA CHAKRA VORTEX: Multi-axis 3D orbital rings & photon sphere event horizon
              const sphereR = Math.max(q1.span, q2.span) * 2.0;
              glow(c, midX, midY, sphereR * 1.7, "rgba(56, 189, 248, 0.65)", "rgba(2, 132, 199, 0)");
              glow(c, midX, midY, sphereR * 0.4, "rgba(255, 255, 255, 1.0)", "rgba(56, 189, 248, 0)");

              c.shadowColor = "#38bdf8";
              c.shadowBlur = 25;
              c.lineWidth = 2.4;

              // 6 High-speed tilted orbital accretion discs
              for (let k = 0; k < 6; k++) {
                c.strokeStyle = (k % 2 === 0) ? "rgba(255, 255, 255, 0.95)" : "rgba(186, 230, 253, 0.85)";
                c.beginPath();
                const tilt = 0.45 + (k % 3) * 0.25;
                for (let u = 0; u <= 1; u += 0.02) {
                  const a = (k * TAU) / 6 + u * 14 - t * 9;
                  const rr = sphereR * (1 - u * 0.85);
                  const x = midX + Math.cos(a) * rr;
                  const y = midY + Math.sin(a) * rr * tilt;
                  u === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
                }
                c.stroke();
              }

              // Inward spiral chakra streams converging into the black-hole event horizon
              for (let i = 0; i < 4; i++) {
                const a = Math.random() * TAU;
                emit(midX + Math.cos(a) * sphereR, midY + Math.sin(a) * sphereR, (Math.random() - 0.5) * 7, (Math.random() - 0.5) * 7, "#7dd3fc", 3.5);
              }
            } else {
              // Dual-core spiral vortex linking both chakra spheres
              c.strokeStyle = "rgba(56, 189, 248, 0.95)";
              c.lineWidth = 2.5;
              c.shadowColor = "#0284c7";
              c.shadowBlur = 16;
              const steps = 28;
              c.beginPath();
              for (let i = 0; i <= steps; i++) {
                const u = i / steps;
                const x = p1.x + (p2.x - p1.x) * u;
                const y = p1.y + (p2.y - p1.y) * u + Math.sin(u * Math.PI * 4 + t * 5.5) * 22;
                i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
              }
              c.stroke();
            }
            c.restore();
          } else {
            c.save();
            if (isClash) {
              // WORMHOLE COLLAPSE: Relativistic gravitational funnel with burning event horizon
              const portalR = Math.max(q1.span, q2.span) * 2.3;
              glow(c, midX, midY, portalR * 1.6, "rgba(249, 115, 22, 0.65)", "rgba(194, 65, 12, 0)");
              glow(c, midX, midY, portalR * 0.45, "rgba(255, 237, 213, 0.98)", "rgba(249, 115, 22, 0)");

              c.strokeStyle = "rgba(254, 215, 170, 0.98)";
              c.lineWidth = 3.2;
              c.shadowColor = "#ea580c";
              c.shadowBlur = 24;

              // 4-armed golden spiral accretion funnel
              for (let arm = 0; arm < 4; arm++) {
                c.beginPath();
                for (let u = 0; u <= 1; u += 0.02) {
                  const a = (arm * TAU) / 4 + u * TAU * 3 + t * 5;
                  const r = portalR * (1 - u * 0.9);
                  const x = midX + Math.cos(a) * r;
                  const y = midY + Math.sin(a) * r;
                  u === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
                }
                c.stroke();
              }

              // Inward sucked relativistic particle streams
              for (let i = 0; i < 4; i++) {
                const a = Math.random() * TAU;
                emit(midX + Math.cos(a) * portalR, midY + Math.sin(a) * portalR, -Math.cos(a) * 4.5, -Math.sin(a) * 4.5, "#fb923c", 3.5);
              }
            } else {
              // Dimensional wormhole bridge between sling portals
              c.strokeStyle = "rgba(249, 115, 22, 0.9)";
              c.lineWidth = 2.8;
              c.shadowColor = "#ea580c";
              c.shadowBlur = 18;
              strokePts(c, bolt(p1.x, p1.y, p2.x, p2.y, Math.min(dist * 0.25, 48), 3));
              if (Math.random() < 0.4) {
                emit(midX, midY, (Math.random() - 0.5) * 6, (Math.random() - 0.5) * 6, "#fb923c", 2.8);
              }
            }
            c.restore();
          }
        }
      }
      for (let i = state.portalSparks.length - 1; i >= 0; i--) {
        const s = state.portalSparks[i];
        s.x += s.vx; s.y += s.vy; s.vy += 0.04; s.life -= 0.03;
        if (s.life <= 0) { state.portalSparks.splice(i, 1); continue; }
        c.globalAlpha = s.life; c.fillStyle = s.color;
        c.beginPath(); c.arc(s.x, s.y, s.size * s.life, 0, TAU); c.fill();
      }
      c.restore();
    }

    /* ---------- PRANA CATCH EDU: curriculum quiz modes (answers fall as orbs, catch the right one) ---------- */
    const rnd = n => Math.floor(Math.random() * n), shuffle = a => a.map(v => [Math.random(), v]).sort((x, y) => x[0] - y[0]).map(v => v[1]);
    const QBANK = {
      CAPITALS: [
        ["Capital of France?","Paris","Lyon|Marseille"],
        ["Capital of Japan?","Tokyo","Kyoto|Osaka"],
        ["Capital of India?","New Delhi","Mumbai|Kolkata"],
        ["Capital of Brazil?","Brasilia","Rio de Janeiro|Sao Paulo"],
        ["Capital of Australia?","Canberra","Sydney|Melbourne"],
        ["Capital of Canada?","Ottawa","Toronto|Vancouver"],
        ["Capital of Egypt?","Cairo","Alexandria|Giza"],
        ["Capital of Kenya?","Nairobi","Mombasa|Kisumu"],
        ["Capital of Germany?","Berlin","Munich|Frankfurt"],
        ["Capital of Italy?","Rome","Milan|Naples"],
        ["Capital of Spain?","Madrid","Barcelona|Valencia"],
        ["Capital of Thailand?","Bangkok","Phuket|Chiang Mai"],
        ["Capital of Turkey?","Ankara","Istanbul|Izmir"],
        ["Capital of Argentina?","Buenos Aires","Cordoba|Rosario"],
        ["Capital of Nepal?","Kathmandu","Pokhara|Lalitpur"],
        ["Capital of Peru?","Lima","Cusco|Arequipa"],
        ["Capital of Norway?","Oslo","Bergen|Trondheim"],
        ["Capital of South Korea?","Seoul","Busan|Incheon"],
        ["Capital of South Africa?","Pretoria","Cape Town|Johannesburg"]
      ],
      PHYSICS: [
        ["SI unit of force?","Newton","Joule|Pascal"],
        ["Speed of light in vacuum (km/s)?","300,000","30,000|3,000,000"],
        ["First law of motion is also known as?","Law of Inertia","Law of Force|Law of Action"],
        ["SI unit of electrical resistance?","Ohm","Volt|Ampere"],
        ["Particles of light are called?","Photons","Electrons|Protons"],
        ["Type of lens that converges light?","Convex","Concave|Cylindrical"],
        ["Speed of sound in air (approx m/s)?","343","1500|3000"],
        ["Bending of light as it passes mediums?","Refraction","Reflection|Diffraction"],
        ["SI unit of frequency?","Hertz","Watt|Tesla"],
        ["Energy of motion is called?","Kinetic","Potential|Thermal"],
        ["Standard gravity on Earth (m/s²)?","9.8","6.7|12.4"],
        ["Mirror used for wide rear view?","Convex","Concave|Plane"]
      ],
      CHEMISTRY: [
        ["Chemical symbol for Gold?","Au","Ag|Fe"],
        ["Chemical symbol for Iron?","Fe","Pb|Cu"],
        ["Chemical symbol for Sodium?","Na","K|So"],
        ["Chemical symbol for Silver?","Ag","Au|Al"],
        ["Chemical symbol for Potassium?","K","P|Pt"],
        ["Chemical symbol for Lead?","Pb","Pl|Ld"],
        ["pH of pure water at 25°C?","7","0|14"],
        ["Most abundant gas in atmosphere?","Nitrogen","Oxygen|Argon"],
        ["Chemical formula for table salt?","NaCl","KCl|NaOH"],
        ["Atomic number of Carbon?","6","12|8"],
        ["Lightest element in periodic table?","Hydrogen","Helium|Lithium"],
        ["Acid found in lemons and limes?","Citric Acid","Acetic Acid|Sulfuric Acid"],
        ["Chemical bond sharing electron pairs?","Covalent","Ionic|Metallic"],
        ["Hardest natural allotrope of carbon?","Diamond","Graphite|Fullerene"]
      ],
      BIOLOGY: [
        ["Powerhouse of the cell?","Mitochondria","Nucleus|Ribosome"],
        ["Plants absorb which gas for photosynthesis?","CO2","O2|N2"],
        ["Gas produced by plants in photosynthesis?","Oxygen","Carbon Dioxide|Nitrogen"],
        ["Chambers in a human heart?","4","2|3"],
        ["Number of bones in an adult body?","206","106|306"],
        ["Normal human body temperature (°C)?","37","35|39"],
        ["Universal red blood cell donor type?","O Negative","AB Positive|A Positive"],
        ["Organ that filters blood to produce urine?","Kidneys","Liver|Pancreas"],
        ["Oxygen-carrying protein in red blood cells?","Hemoglobin","Insulin|Collagen"],
        ["Master gland of endocrine system?","Pituitary","Thyroid|Adrenal"],
        ["Molecule storing genetic code?","DNA","RNA|ATP"],
        ["Largest organ of the human body?","Skin","Liver|Lungs"]
      ],
      ASTRONOMY: [
        ["Closest star to Earth?","The Sun","Proxima Centauri|Sirius"],
        ["Largest planet in our solar system?","Jupiter","Saturn|Neptune"],
        ["The Red Planet?","Mars","Venus|Mercury"],
        ["Hottest planet in our solar system?","Venus","Mercury|Mars"],
        ["Planet known for prominent ring system?","Saturn","Uranus|Jupiter"],
        ["Closest galaxy to the Milky Way?","Andromeda","Triangulum|Sombrero"],
        ["Time for sunlight to reach Earth?","8 minutes","8 seconds|8 hours"],
        ["Natural satellite of Earth?","The Moon","Titan|Phobos"],
        ["Largest moon of Saturn?","Titan","Europa|Ganymede"],
        ["Planet tilted on its side at ~98 degrees?","Uranus","Neptune|Mars"]
      ],
      CUSTOM: []
    };
    QBANK.ELEMENTS = QBANK.CHEMISTRY;
    QBANK.SCIENCE = QBANK.PHYSICS;

    function nextQ() {
      const m = state.gmode; let q;
      if (m === "MATH") {
        const mx = 6 + state.level * 4; let a = 1 + rnd(mx), b = 1 + rnd(mx);
        const ops = state.level < 2 ? ["+", "-"] : ["+", "-", "×", "÷"];
        const op = ops[rnd(ops.length)];
        let ans;
        if (op === "+") { ans = a + b; }
        else if (op === "-") { if (b > a) [a, b] = [b, a]; ans = a - b; }
        else if (op === "×") { a = 2 + rnd(3 + state.level * 2); b = 2 + rnd(9); ans = a * b; }
        else { const mult = 2 + rnd(9); b = 2 + rnd(6); a = mult * b; ans = mult; }
        const w = new Set();
        while (w.size < 2) { const v = ans + (rnd(2) ? 1 : -1) * (1 + rnd(4 + state.level)); if (v !== ans && v >= 0) w.add(v); }
        q = { t: `${a} ${op} ${b} = ?`, a: String(ans), w: [...w].map(String), x: `${a} ${op} ${b} = ${ans}` };
      } else {
        const list = (QBANK[m] && QBANK[m].length) ? QBANK[m] : QBANK.PHYSICS;
        const e = list[rnd(list.length)];
        const w = e[2] ? e[2].split("|") : shuffle(list.map(v => v[1]).filter(v => v !== e[1])).slice(0, 2);
        q = { t: e[0], a: e[1], w, x: `${e[0].replace("?", "")}: ${e[1]}` };
      }
      state.quiz = q;
      const opts = shuffle([[q.a, 1], ...q.w.map(v => [v, 0])]), cols = ["#38bdf8", "#ec4899", "#a855f7"], W = vecCanvas.width;
      opts.forEach(([label, ok], i) => state.gems.push({ x: W * (0.2 + i * 0.3) + (Math.random() - .5) * 30, y: -30 - rnd(60), r: Math.max(36, label.length * 5.4 + 16), vx: 0,
        vy: 1.3 + state.level * 0.15, type: "ans", ok, label, ph: Math.random() * 6, color: cols[i] }));
    }
    function setFb(t, good) { state.fb = t; state.fbGood = good; state.fbT = 120; }
    /* ---------- PRANA CATCH 2.0: orbs, gold bonus, bombs, combo, levels, lives, magnet mouth ---------- */
    function resetGame() { Object.assign(state, { gameScore: 0, gems: [], particles: [], rings: [], lives: 3, streak: 0, level: 1, over: false, flash: 0, quiz: null, fb: "", fbT: 0 }); }
    function spawnGem() {
      const r = Math.random(), type = r < 0.16 ? "bomb" : r < 0.24 ? "gold" : "orb";
      state.gems.push({ x: Math.random() * (vecCanvas.width - 80) + 40, y: -30, r: type === "gold" ? 18 : 22, vx: 0,
        vy: (2 + Math.random() * 1.8) * (1 + state.level * 0.12), type, ph: Math.random() * 6,
        color: type === "gold" ? "#fbbf24" : ["#38bdf8", "#ec4899", "#a855f7", "#34d399"][Math.floor(Math.random() * 4)] });
    }
    function ring(x, y, color) { state.rings.push({ x, y, r: 10, life: 1, color }); }
    function drawGame(w, h) {
      const c = vCtx, t = performance.now() / 1000;
      if (state.gameGalaxy) {
        c.save();
        c.globalCompositeOperation = "source-over";
        drawGalaxyBackground(c, w, h, t);
        c.restore();
      }
      const head = state.faceLandmarks ? landmarkToScreen(state.faceLandmarks[1]) : null;
      const hands = state.handData.map(hd => landmarkToScreen(hd.landmarks[8]));
      const open = state.jawOpen > 0.4, mult = Math.min(5, 1 + Math.floor(state.streak / 5));
      if (state.over) {
        if (state.jawOpen > 0.55) resetGame();
        badge(`GAME OVER · ${state.gameScore} · BEST ${state.best} · OPEN MOUTH TO RESTART`);
      } else {
        state.level = 1 + Math.floor(state.gameScore / 150);
        if (state.gmode === "ARCADE") { if (Math.random() < 0.02 + state.level * 0.004 && state.gems.length < 4 + state.level) spawnGem(); }
        else if (!state.gems.some(g => g.type === "ans")) nextQ();
        badge(`${state.gmode === "ARCADE" ? "PRANA" : state.gmode} ${state.gameScore} · x${mult} · LV ${state.level} · LIVES: ${state.lives}`);
      }
      c.save();
      if (state.flash > 0) { c.fillStyle = `rgba(255,30,70,${state.flash * 0.25})`; c.fillRect(0, 0, w, h); state.flash -= 0.06; }
      if (state.gmode !== "ARCADE" && state.quiz) {
        c.save();
        c.globalCompositeOperation = "source-over";
        const bw = Math.min(w * 0.92, 580), bx = w / 2 - bw / 2;
        const by = 104;
        const qText = state.quiz.t || "";

        // Dynamic font sizing & multi-line wrapping so text never spills out of gray card
        let fontSize = 20;
        c.font = `700 ${fontSize}px system-ui, -apple-system, sans-serif`;
        const maxTextWidth = bw - 36;
        let textWidth = c.measureText(qText).width;

        if (textWidth > maxTextWidth) {
          fontSize = Math.max(13, Math.floor(fontSize * (maxTextWidth / textWidth)));
          c.font = `700 ${fontSize}px system-ui, -apple-system, sans-serif`;
          textWidth = c.measureText(qText).width;
        }

        let lines = [qText];
        if (textWidth > maxTextWidth) {
          const words = qText.split(" ");
          lines = [];
          let cur = "";
          for (const word of words) {
            const test = cur ? cur + " " + word : word;
            if (c.measureText(test).width <= maxTextWidth) {
              cur = test;
            } else {
              if (cur) lines.push(cur);
              cur = word;
            }
          }
          if (cur) lines.push(cur);
          fontSize = 13;
          c.font = `700 ${fontSize}px system-ui, -apple-system, sans-serif`;
        }

        const boxH = lines.length > 1 ? 64 : 50;

        // Card Container Box (non-overlapping below status badge)
        c.fillStyle = "rgba(10, 14, 23, 0.88)";
        c.strokeStyle = "rgba(56, 189, 248, 0.55)";
        c.lineWidth = 1.5;
        c.beginPath();
        c.roundRect(bx, by, bw, boxH, 12);
        c.fill();
        c.stroke();

        // Card Question Text
        c.fillStyle = "#f8fafc";
        c.textAlign = "center";
        c.textBaseline = "middle";
        if (lines.length === 1) {
          c.fillText(lines[0], w / 2, by + boxH / 2);
        } else {
          const lineGap = fontSize * 1.25;
          const startY = by + boxH / 2 - ((lines.length - 1) * lineGap) / 2;
          lines.forEach((line, li) => {
            c.fillText(line, w / 2, startY + li * lineGap);
          });
        }

        // Distinct Non-Overlapping Feedback Pill
        if (state.fbT > 0) {
          state.fbT--;
          c.globalAlpha = Math.min(1, state.fbT / 25);
          const fbText = state.fb || "";
          c.font = "600 13px system-ui, -apple-system, sans-serif";
          const fbw = Math.min(bw, c.measureText(fbText).width + 32);
          const fbh = 28;
          const fby = by + boxH + 8;
          const fbx = w / 2 - fbw / 2;

          c.fillStyle = state.fbGood ? "rgba(6, 78, 59, 0.90)" : "rgba(136, 19, 55, 0.90)";
          c.strokeStyle = state.fbGood ? "rgba(52, 211, 153, 0.7)" : "rgba(251, 113, 133, 0.7)";
          c.lineWidth = 1;
          c.beginPath();
          c.roundRect(fbx, fby, fbw, fbh, 14);
          c.fill();
          c.stroke();

          c.fillStyle = state.fbGood ? "#a7f3d0" : "#fecdd3";
          c.fillText(fbText, w / 2, fby + fbh / 2);
        }
        c.restore();
      }
      const targets = [head, ...hands].filter(Boolean);
      for (const [ptrId, ptr] of activePointers.entries()) {
        if (ptr.isDown) targets.push({ x: ptr.x, y: ptr.y, isPointer: true });
      }
      for (let i = state.gems.length - 1; i >= 0; i--) {
        const g = state.gems[i];
        if (g.dead) { state.gems.splice(i, 1); continue; }
        if (!state.over) {
          g.y += g.vy; g.x += g.vx;
          let hit = false;
          for (const tg of targets) {
            const d = Math.hypot(g.x - tg.x, g.y - tg.y);
            if (open && tg === head && d < 170 && g.type !== "bomb") { g.vx += (tg.x - g.x) * 0.004; g.vy += (tg.y - g.y) * 0.004; }
            if (d < g.r + (tg === head ? (open ? 46 : 26) : 30)) { hit = true; break; }
          }
          if (hit) {
            if (g.type === "bomb") {
              state.lives--; state.streak = 0; state.flash = 1; ring(g.x, g.y, "#ff3355"); playAudioTone(110, "sawtooth", 0.4, 0.25);
              if (state.lives <= 0) { state.over = true; if (state.gameScore > state.best) { state.best = state.gameScore; try { localStorage.setItem("divya_best", state.best); } catch (e) {} } }
            } else if (g.type === "ans") {
              if (g.ok) {
                state.streak++; state.gameScore += 20 * Math.min(5, 1 + Math.floor(state.streak / 5));
                playAudioTone(660 + Math.min(state.streak, 20) * 25, "triangle", 0.25); ring(g.x, g.y, "#34d399");
                setFb("CORRECT · " + state.quiz.x, true); state.gems.forEach(o => { if (o.type === "ans") o.dead = 1; });
              } else {
                state.lives--; state.streak = 0; state.flash = 1; ring(g.x, g.y, "#ff3355"); playAudioTone(140, "sawtooth", 0.3, 0.2);
                setFb("INCORRECT · " + state.quiz.x, false);
                if (state.lives <= 0) { state.over = true; if (state.gameScore > state.best) { state.best = state.gameScore; try { localStorage.setItem("divya_best", state.best); } catch (e) {} } }
              }
            } else {
              state.streak++;
              state.gameScore += (g.type === "gold" ? 30 : 10) * Math.min(5, 1 + Math.floor(state.streak / 5));
              playAudioTone(520 + Math.min(state.streak, 20) * 30 + Math.random() * 120, "triangle", 0.22);
              ring(g.x, g.y, g.color);
              for (let k = 0; k < 14; k++) { const a = Math.random() * TAU, s = 2.5 + Math.random() * 4.5; state.particles.push({ x: g.x, y: g.y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, life: 1, color: g.color }); }
            }
            state.gems.splice(i, 1); continue;
          }
          if (g.y > h + 40) {
            if (g.type === "ans") { if (g.ok) { state.streak = 0; setFb("Missed! " + state.quiz.x, false); state.gems.forEach(o => { if (o.type === "ans") o.dead = 1; }); } }
            else if (g.type !== "bomb") state.streak = 0;
            state.gems.splice(i, 1); continue;
          }
        }
        const pu = 1 + 0.08 * Math.sin(t * 4 + g.ph);
        if (g.type === "bomb") {
          c.globalCompositeOperation = "source-over";
          glow(c, g.x, g.y, g.r * 1.8, `rgba(255,30,70,${0.25 + 0.2 * Math.sin(t * 8)})`, "rgba(255,30,70,0)");
          c.fillStyle = "#12060b"; c.strokeStyle = "#ff3355"; c.lineWidth = 2;
          c.beginPath(); c.arc(g.x, g.y, g.r, 0, TAU); c.fill(); c.stroke();
          for (let s = 0; s < 8; s++) { const a = s * TAU / 8 + t; c.beginPath(); c.moveTo(g.x + Math.cos(a) * g.r, g.y + Math.sin(a) * g.r); c.lineTo(g.x + Math.cos(a) * (g.r + 7), g.y + Math.sin(a) * (g.r + 7)); c.stroke(); }
        } else {
          c.globalCompositeOperation = "lighter";
          const tg = c.createLinearGradient(g.x, g.y - g.vy * 9, g.x, g.y);
          tg.addColorStop(0, g.color + "00"); tg.addColorStop(1, g.color + "aa");
          c.strokeStyle = tg; c.lineWidth = g.r * 0.9; c.lineCap = "round";
          c.beginPath(); c.moveTo(g.x, g.y - g.vy * 9); c.lineTo(g.x, g.y); c.stroke();
          glow(c, g.x, g.y, g.r * 2.4 * pu, g.color + "88", g.color + "00");
          if (g.type === "ans") {
            glow(c, g.x, g.y, g.r * pu, "rgba(8,12,30,0.88)", g.color);
            c.globalCompositeOperation = "source-over"; c.fillStyle = "#fff"; c.font = "bold 17px system-ui, sans-serif"; c.textAlign = "center"; c.textBaseline = "middle";
            c.fillText(g.label, g.x, g.y); c.globalCompositeOperation = "lighter";
          } else glow(c, g.x, g.y, g.r * pu, "#ffffff", g.color);
        }
      }
      c.globalCompositeOperation = "lighter";
      for (let i = state.rings.length - 1; i >= 0; i--) {
        const r = state.rings[i]; r.r += 6; r.life -= 0.04;
        if (r.life <= 0) { state.rings.splice(i, 1); continue; }
        c.strokeStyle = r.color; c.globalAlpha = r.life; c.lineWidth = 3 * r.life; c.beginPath(); c.arc(r.x, r.y, r.r, 0, TAU); c.stroke();
      }
      for (let i = state.particles.length - 1; i >= 0; i--) {
        const p = state.particles[i]; p.x += p.vx; p.y += p.vy; p.vy += 0.22; p.life -= 0.035;
        if (p.life <= 0) { state.particles.splice(i, 1); continue; }
        c.globalAlpha = p.life; c.fillStyle = p.color; c.beginPath(); c.arc(p.x, p.y, 4 * p.life, 0, TAU); c.fill();
      }
      c.globalAlpha = 1;
      if (head) {
        state.headTrail.push({ x: head.x, y: head.y }); if (state.headTrail.length > 18) state.headTrail.shift();
        state.headTrail.forEach((p, i) => { c.fillStyle = `rgba(56,189,248,${i / 40})`; c.beginPath(); c.arc(p.x, p.y, 2 + i * 0.4, 0, TAU); c.fill(); });
        c.strokeStyle = "#38bdf8"; c.lineWidth = 2; c.shadowColor = "#38bdf8"; c.shadowBlur = 12;
        c.beginPath(); c.arc(head.x, head.y, (open ? 34 : 18) + Math.sin(t * 6) * 2, 0, TAU); c.stroke();
        if (open) { c.setLineDash([6, 10]); c.lineDashOffset = -t * 40; c.globalAlpha = 0.5; c.beginPath(); c.arc(head.x, head.y, 170, 0, TAU); c.stroke(); }
      }
      hands.forEach(p => { c.strokeStyle = "#a855f7"; c.lineWidth = 2; c.beginPath(); c.arc(p.x, p.y, 16, 0, TAU); c.stroke(); });
      c.restore();
    }


    /* =========================================================================
       AVATAR: auto-rigs a static GLB (builds a humanoid skeleton + skin weights), then drives it from MediaPipe pose.
       ========================================================================= */
    const avCanvas = document.getElementById("avatar-canvas");
    const AV = { ready: false, loading: false, T: null, renderer: null, scene: null, cam: null, rig: null, specs: [], byN: {}, q: {}, s: 300, cal: null, raw: {}, bg: 0, msgUntil: 0 };
    // Joint positions of the bundled model (A-pose, Y-up, facing +Z, feet at y=0), measured from the mesh. Avatar "L" = +X side.
    const JS = { hips: [0, .93, .04], spine: [0, 1.1, .05], chest: [0, 1.28, .05], neck: [0, 1.47, .04], head: [0, 1.58, .04], headTop: [0, 1.85, .04],
      sh: [.17, 1.37, .03], el: [.27, 1.13, .05], wr: [.39, 1.01, .10], hand: [.45, .98, .12], hp: [.09, .92, .04], kn: [.12, .5, .06], an: [.16, .08, .04], toe: [.17, .03, .17] };
    const AV_BG = ["radial-gradient(ellipse at 50% 35%,#1b2340 0%,#06070a 75%)", "repeating-linear-gradient(0deg,rgba(56,189,248,.16) 0 1px,transparent 1px 48px),repeating-linear-gradient(90deg,rgba(56,189,248,.16) 0 1px,transparent 1px 48px),#070a18", "#000", "#00b140"];
    async function avatarStart(src) {
      if (AV.loading) return; AV.loading = true; badge("LOADING AVATAR…");
      try {
        const T = AV.T = await import("three"); const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
        if (!AV.renderer) {
          AV.renderer = new T.WebGLRenderer({ canvas: avCanvas, alpha: true, antialias: true, preserveDrawingBuffer: true }); AV.renderer.setClearColor(0x000000, 0);
          AV.scene = new T.Scene(); AV.scene.add(new T.HemisphereLight(0xffffff, 0x445566, 1.6));
          const dl = new T.DirectionalLight(0xffffff, 2.2); dl.position.set(200, 500, 900); AV.scene.add(dl);
          AV.cam = new T.OrthographicCamera(-1, 1, 1, -1, -5000, 5000);
        }
        const loader = new GLTFLoader();
        let gltf = null;
        const targetSrc = src || AV.currentModel || "./assets/avatar2.glb"; AV.currentModel = targetSrc;
        try {
          gltf = await loader.loadAsync(targetSrc);
        } catch (err1) {
          if (!src || src.includes("avatar")) {
            const fallbackSrc = targetSrc.includes("avatar2") ? "./assets/avatar.glb" : "./assets/avatar2.glb";
            gltf = await loader.loadAsync(fallbackSrc);
          } else {
            throw err1;
          }
        }
        avatarBuild(gltf.scene); AV.ready = true;
      } catch (e) { console.error(e); badge("AVATAR FAILED · verify avatar.glb is present and open via https/localhost"); AV.msgUntil = performance.now() + 6000; }
      AV.loading = false;
    }
    function avatarBuild(root) {
      const T = AV.T; root.updateMatrixWorld(true);
      if (AV.rig) AV.scene.remove(AV.rig);

      // Check if model has a pre-existing skeleton / skinned meshes (e.g. avatar2.glb / Freyja)
      const existingBones = [];
      root.traverse(o => { if (o.isBone) existingBones.push(o); });

      if (existingBones.length > 0) {
        // --- Skinned Mesh Rig (avatar2.glb / Freyja 118-bone rig) ---
        const box = new T.Box3().setFromObject(root);
        const cx = (box.min.x + box.max.x) / 2;
        const cy = box.min.y;
        const cz = (box.min.z + box.max.z) / 2;
        const modelHeight = Math.max(0.1, box.max.y - box.min.y);

        // Normalize root pivot: center horizontally (X=0, Z=0) and ground feet to y=0
        root.position.set(-cx, -cy, -cz);

        const rig = new T.Group();
        rig.add(root);
        rig.updateMatrixWorld(true);

        const findBone = (rgx) => {
          let found = null;
          root.traverse(o => {
            if (!found && o.isBone && rgx.test(o.name)) found = o;
          });
          return found;
        };

        const bones = {
          hips: findBone(/hips/i),
          spine: findBone(/spine/i),
          chest: findBone(/chest/i),
          neck: findBone(/neck/i),
          head: findBone(/head/i),
          shoulderL: findBone(/shoulder.*l/i),
          shoulderR: findBone(/shoulder.*r/i),
          uArmL: findBone(/arm.*l/i),
          uArmR: findBone(/arm.*r/i),
          lArmL: findBone(/elbow.*l/i),
          lArmR: findBone(/elbow.*r/i),
          handL: findBone(/hand.*l(?!.*(?:thumb|index|middle|ring|pinky))/i),
          handR: findBone(/hand.*r(?!.*(?:thumb|index|middle|ring|pinky))/i),
          uLegL: findBone(/leg.*l/i),
          uLegR: findBone(/leg.*r/i),
          lLegL: findBone(/knee.*l/i),
          lLegR: findBone(/knee.*r/i),
          footL: findBone(/foot.*l/i),
          footR: findBone(/foot.*r/i)
        };

        // Cache initial rest quaternions for stable relative transformations
        Object.values(bones).forEach(b => {
          if (b) b.userData.initQuat = b.quaternion.clone();
        });

        // Determine rest positions for screen anchoring
        const hipWorld = new T.Vector3();
        if (bones.hips) bones.hips.getWorldPosition(hipWorld);
        const chestWorld = new T.Vector3();
        if (bones.chest) bones.chest.getWorldPosition(chestWorld);

        const hipRest = new T.Vector3(0, hipWorld.y > 0 ? hipWorld.y - cy : modelHeight * 0.52, 0);
        const shRest = new T.Vector3(0, chestWorld.y > 0 ? chestWorld.y - cy : modelHeight * 0.74, 0);

        AV.scene.add(rig);
        Object.assign(AV, {
          rig,
          isSkinned: true,
          bones,
          specs: [],
          byN: {},
          q: {},
          hipRest,
          shRest,
          modelHeight
        });
      } else {
        // --- Static Unrigged Mesh (avatar.glb / Pandora procedural 16-bone auto-rigger) ---
        AV.isSkinned = false;
        const meshes = []; root.traverse(o => { if (o.isMesh) meshes.push(o); });
        const box = new T.Box3().setFromObject(root), k = (box.max.y - box.min.y) / 1.85, cx = (box.min.x + box.max.x) / 2, base = box.min.y;
        const J = {}; for (const key in JS) J[key] = new T.Vector3(JS[key][0] * k + cx, JS[key][1] * k + base, JS[key][2] * k);
        const specs = [], add = (n, p, a, b) => specs.push({ n, p, a, b });
        add("hips", null, J.hips, J.spine); add("spine", "hips", J.spine, J.chest); add("chest", "spine", J.chest, J.neck); add("neck", "chest", J.neck, J.head); add("head", "neck", J.head, J.headTop);
        [["L", 1], ["R", -1]].forEach(([S, sg]) => {
          const v = q => new T.Vector3((J[q].x - cx) * sg + cx, J[q].y, J[q].z);
          add("uArm" + S, "chest", v("sh"), v("el")); add("lArm" + S, "uArm" + S, v("el"), v("wr")); add("hand" + S, "lArm" + S, v("wr"), v("hand"));
          add("uLeg" + S, "hips", v("hp"), v("kn")); add("lLeg" + S, "uLeg" + S, v("kn"), v("an")); add("foot" + S, "lLeg" + S, v("an"), v("toe"));
        });
        const byN = {}, bones = specs.map(sp => { const b = new T.Bone(); b.name = sp.n; return b; });
        specs.forEach((sp, i) => {
          byN[sp.n] = sp; sp.bone = bones[i]; const par = sp.p ? byN[sp.p] : null;
          bones[i].position.copy(sp.a).sub(par ? par.a : new T.Vector3()); if (par) par.bone.add(bones[i]);
          sp.rest = sp.b.clone().sub(sp.a).normalize();
        });
        const rig = new T.Group(); rig.add(bones[0]); rig.updateMatrixWorld(true);
        const skeleton = new T.Skeleton(bones), shX = J.sh.x - cx;
        meshes.forEach(mesh => {
          const geom = mesh.geometry.clone(); geom.applyMatrix4(mesh.matrixWorld);
          const pos = geom.attributes.position, n = pos.count, si = new Uint16Array(n * 4), sw = new Float32Array(n * 4), hair = /hair/i.test(mesh.name + " " + (mesh.material && mesh.material.name || ""));
          for (let v = 0; v < n; v++) {
            const px = pos.getX(v), py = pos.getY(v), pz = pos.getZ(v), cand = [];
            specs.forEach((sp, bi) => {
              const nm = sp.n; if (hair && !/^(spine|chest|neck|head)$/.test(nm)) return;
              const ax = sp.a.x, ay = sp.a.y, az = sp.a.z, bx = sp.b.x - ax, by = sp.b.y - ay, bz = sp.b.z - az;
              let u = ((px - ax) * bx + (py - ay) * by + (pz - az) * bz) / (bx * bx + by * by + bz * bz); u = u < 0 ? 0 : u > 1 ? 1 : u;
              let d = Math.hypot(px - ax - bx * u, py - ay - by * u, pz - az - bz * u);
              const side = /L$/.test(nm) ? 1 : /R$/.test(nm) ? -1 : 0;
              if (side) { const rx = (px - cx) * side; if (rx < 0) d += 1; else if (/Arm|hand/.test(nm)) d += Math.max(0, shX + 0.02 * k - rx) * 4; }
              cand.push([d, bi]);
            });
            cand.sort((a, b) => a[0] - b[0]); let tot = 0; const ws = [];
            for (let q = 0; q < 4; q++) { const w = 1 / Math.pow(cand[q][0] + 0.015, 3); ws.push(w); tot += w; }
            for (let q = 0; q < 4; q++) { si[v * 4 + q] = cand[q][1]; sw[v * 4 + q] = ws[q] / tot; }
          }
          geom.setAttribute("skinIndex", new T.Uint16BufferAttribute(si, 4)); geom.setAttribute("skinWeight", new T.Float32BufferAttribute(sw, 4));
          const m = mesh.material; if (m) { m.side = T.DoubleSide; if (m.metalness > 0.3) m.metalness = 0.3; if (hair) { m.transparent = true; m.alphaTest = 0.35; } }
          const sm = new T.SkinnedMesh(geom, m); sm.frustumCulled = false; rig.add(sm); sm.bind(skeleton, new T.Matrix4());
        });
        const shRest = new T.Vector3(cx, J.sh.y, J.sh.z);
        AV.scene.add(rig); Object.assign(AV, { rig, specs, byN, q: {}, hipRest: J.hips.clone(), shRest, modelHeight: 1.85 });
      }
    }
    function avCalibrate() {
      let n = 3; const tick = () => {
        AV.msgUntil = performance.now() + 1100;
        if (n > 0) { badge(`CALIBRATING · hold posture · ${n}`); n--; setTimeout(tick, 1000); }
        else if (AV.raw.chest) {
          AV.cal = {
            hips: AV.raw.hips ? AV.raw.hips.clone() : new AV.T.Quaternion(),
            chest: AV.raw.chest.clone(),
            head: AV.raw.head.clone()
          };
          badge("CALIBRATED ✓");
          AV.msgUntil = performance.now() + 1500;
          playAudioTone(880, "triangle", 0.2);
        }
      }; tick();
    }
    function avatarFrame() {
      if (!AV.ready) return;
      const W = vecCanvas.width, H = vecCanvas.height;
      if (AV.renderer.domElement.width !== W || AV.renderer.domElement.height !== H) { AV.renderer.setPixelRatio(1); AV.renderer.setSize(W, H, false); Object.assign(AV.cam, { left: -W / 2, right: W / 2, top: H / 2, bottom: -H / 2 }); AV.cam.updateProjectionMatrix(); }
      const P = state.poseLandmarks, Wd = state.poseWorld;
      if (P && Wd) avatarDrive(P, Wd, W, H);
      if (performance.now() > AV.msgUntil) badge(P ? (AV.cal ? "AVATAR · calibrated · mirror mode" : "AVATAR · tap Calibrate for a perfect match") : "AVATAR · step back so your upper body is in frame");
      AV.renderer.render(AV.scene, AV.cam);
    }
    function avatarDrive(P, Wd, W, H) {
      const T = AV.T, mir = state.mirrorView;

      // Temporal Exponential Moving Average (EMA) Landmark Smoothing (Kalidokit / VRM filter standard)
      if (!AV.smoothP || AV.smoothP.length !== P.length) {
        AV.smoothP = P.map(p => ({ x: p.x, y: p.y, z: p.z, visibility: p.visibility }));
        AV.smoothWd = Wd.map(p => ({ x: p.x, y: p.y, z: p.z }));
      } else {
        const pAlpha = 0.28; // Smooth low-pass factor eliminating camera sensor jitter
        for (let i = 0; i < P.length; i++) {
          AV.smoothP[i].x += (P[i].x - AV.smoothP[i].x) * pAlpha;
          AV.smoothP[i].y += (P[i].y - AV.smoothP[i].y) * pAlpha;
          AV.smoothP[i].z += (P[i].z - AV.smoothP[i].z) * pAlpha;
          AV.smoothP[i].visibility = P[i].visibility;
          AV.smoothWd[i].x += (Wd[i].x - AV.smoothWd[i].x) * pAlpha;
          AV.smoothWd[i].y += (Wd[i].y - AV.smoothWd[i].y) * pAlpha;
          AV.smoothWd[i].z += (Wd[i].z - AV.smoothWd[i].z) * pAlpha;
        }
      }
      const sP = AV.smoothP, sWd = AV.smoothWd;
      const xs = Math.sign((sP[11].x - sP[12].x) * (sWd[11].x - sWd[12].x)) || 1, mx = (mir ? -1 : 1) * xs;
      const S = i => new T.Vector3(mx * sWd[i].x, -sWd[i].y, -sWd[i].z), vis = i => sP[i].visibility == null ? 1 : sP[i].visibility;
      const ix = (b, side) => b + ((side === "L") === !mir ? 0 : 1), mid = (a, b) => a.clone().add(b).multiplyScalar(0.5);
      const basis = (r, u) => { r = r.clone().normalize(); const f = new T.Vector3().crossVectors(r, u).normalize(), u2 = new T.Vector3().crossVectors(f, r); return new T.Quaternion().setFromRotationMatrix(new T.Matrix4().makeBasis(r, u2, f)); };

      // Hysteresis on hips visibility to eliminate sudden frame-to-frame seated/standing anchor jumping
      const rawHipsVis = vis(23) > 0.45 && vis(24) > 0.45 && sP[23].y < 0.90 && sP[24].y < 0.90;
      if (AV.hipsVis == null) AV.hipsVis = rawHipsVis;
      else if (AV.hipsVis && (!rawHipsVis && (vis(23) < 0.35 || vis(24) < 0.35))) AV.hipsVis = false;
      else if (!AV.hipsVis && (rawHipsVis && (vis(23) > 0.55 && vis(24) > 0.55))) AV.hipsVis = true;
      const hipsVis = AV.hipsVis;

      const hipL = S(ix(23, "L")), hipR = S(ix(23, "R")), shL = S(ix(11, "L")), shR = S(ix(11, "R"));
      const up = hipsVis ? mid(shL, shR).sub(mid(hipL, hipR)) : new T.Vector3(0, 1, 0);

      const earL = S(ix(7, "L")), earR = S(ix(7, "R")), face = S(0).sub(mid(earL, earR)); face.y += face.length() * 0.33;
      const uH = new T.Vector3().crossVectors(face, earL.clone().sub(earR)).normalize(), rH = new T.Vector3().crossVectors(uH, face).normalize();

      const rawHips = hipsVis ? basis(hipL.clone().sub(hipR), up) : basis(shL.clone().sub(shR), up);
      const rawChest = basis(shL.clone().sub(shR), up);
      const rawHead = new T.Quaternion().setFromRotationMatrix(new T.Matrix4().makeBasis(rH, uH, face.clone().normalize()));

      const raw = AV.raw = { hips: rawHips, chest: rawChest, head: rawHead };
      const rel = k => AV.cal ? raw[k].clone().multiply(AV.cal[k].clone().invert()) : raw[k].clone(), tgt = {};
      tgt.hips = rel("hips"); tgt.chest = rel("chest"); tgt.head = rel("head");
      tgt.spine = tgt.hips.clone().slerp(tgt.chest, 0.5); tgt.neck = tgt.chest.clone().slerp(tgt.head, 0.5);
      const defaultRests = {
        uArmL: new T.Vector3(0.5, -0.86, 0).normalize(),
        uArmR: new T.Vector3(-0.5, -0.86, 0).normalize(),
        lArmL: new T.Vector3(0, -1, 0),
        lArmR: new T.Vector3(0, -1, 0),
        handL: new T.Vector3(0, -1, 0),
        handR: new T.Vector3(0, -1, 0),
        uLegL: new T.Vector3(0, -1, 0),
        uLegR: new T.Vector3(0, -1, 0),
        lLegL: new T.Vector3(0, -1, 0),
        lLegR: new T.Vector3(0, -1, 0),
        footL: new T.Vector3(0, 0, 1),
        footR: new T.Vector3(0, 0, 1)
      };
      const swing = (name, a, b, v) => {
        const d = b.clone().sub(a);
        const rest = (AV.byN && AV.byN[name] && AV.byN[name].rest) || defaultRests[name] || new T.Vector3(0, -1, 0);
        tgt[name] = (v < 0.4 || d.lengthSq() < 1e-6) ? new T.Quaternion() : new T.Quaternion().setFromUnitVectors(rest, d.normalize());
      };
      ["L", "R"].forEach(sd => {
        const a = q => ix(q, sd), mv = (...q) => Math.min(...q.map(vis));
        swing("uArm" + sd, S(a(11)), S(a(13)), mv(a(11), a(13))); swing("lArm" + sd, S(a(13)), S(a(15)), mv(a(13), a(15)));
        swing("hand" + sd, S(a(15)), mid(S(a(17)), S(a(19))), mv(a(15), a(19)));
        if (hipsVis) {
          swing("uLeg" + sd, S(a(23)), S(a(25)), mv(a(23), a(25))); swing("lLeg" + sd, S(a(25)), S(a(27)), mv(a(25), a(27))); swing("foot" + sd, S(a(27)), S(a(31)), mv(a(27), a(31)));
        } else {
          tgt["uLeg" + sd] = new T.Quaternion();
          tgt["lLeg" + sd] = new T.Quaternion();
          tgt["foot" + sd] = new T.Quaternion();
        }
      });
      if (AV.isSkinned && AV.bones) {
        const applyRot = (bone, targetQ, lerpRate = 0.22) => {
          if (!bone || !targetQ) return;
          const key = bone.name;
          const q = AV.q[key] || (AV.q[key] = new T.Quaternion());
          q.slerp(targetQ, lerpRate);
          bone.quaternion.copy(bone.userData.initQuat).multiply(q);
        };
        applyRot(AV.bones.hips, tgt.hips, 0.20);
        applyRot(AV.bones.spine, tgt.spine, 0.20);
        applyRot(AV.bones.chest, tgt.chest, 0.22);
        applyRot(AV.bones.neck, tgt.neck, 0.22);
        applyRot(AV.bones.head, tgt.head, 0.25);

        ["L", "R"].forEach(sd => {
          applyRot(AV.bones["uArm" + sd], tgt["uArm" + sd], 0.24);
          applyRot(AV.bones["lArm" + sd], tgt["lArm" + sd], 0.24);
          applyRot(AV.bones["hand" + sd], tgt["hand" + sd], 0.25);
          if (hipsVis) {
            applyRot(AV.bones["uLeg" + sd], tgt["uLeg" + sd], 0.18);
            applyRot(AV.bones["lLeg" + sd], tgt["lLeg" + sd], 0.18);
            applyRot(AV.bones["foot" + sd], tgt["foot" + sd], 0.18);
          } else if (AV.bones["uLeg" + sd]) {
            AV.bones["uLeg" + sd].quaternion.copy(AV.bones["uLeg" + sd].userData.initQuat);
            if (AV.bones["lLeg" + sd]) AV.bones["lLeg" + sd].quaternion.copy(AV.bones["lLeg" + sd].userData.initQuat);
            if (AV.bones["foot" + sd]) AV.bones["foot" + sd].quaternion.copy(AV.bones["foot" + sd].userData.initQuat);
          }
        });
      } else {
        AV.specs.forEach(sp => {
          const q = AV.q[sp.n] || (AV.q[sp.n] = new T.Quaternion()); q.slerp(tgt[sp.n], /Leg|foot/.test(sp.n) ? 0.18 : 0.25);
          sp.bone.quaternion.copy(sp.p ? AV.q[sp.p].clone().invert().multiply(q) : q);
        });
      }
      // Damped screen-space positioning & scale
      const avg = (i, j) => ({ x: (sP[i].x + sP[j].x) / 2, y: (sP[i].y + sP[j].y) / 2, z: 0 });
      const sh = landmarkToScreen(avg(11, 12)), a = landmarkToScreen(sP[11]), b = landmarkToScreen(sP[12]);
      const shoulderDist = Math.hypot(a.x - b.x, a.y - b.y);

      let targetScale;
      if (hipsVis) {
        const hp = landmarkToScreen(avg(23, 24));
        targetScale = Math.max(Math.hypot(sh.x - hp.x, sh.y - hp.y) / 0.44, shoulderDist / 0.34);
      } else {
        targetScale = shoulderDist > 10 ? (shoulderDist / 0.34) : (H * 0.65);
      }
      if (isFinite(targetScale) && targetScale > 20) {
        AV.s += (Math.min(targetScale, H * 1.5) - AV.s) * 0.12;
        AV.rig.scale.setScalar(AV.s);
      }

      if (!AV.pos) AV.pos = new T.Vector3();
      let targetX, targetY;
      if (hipsVis) {
        const hp = landmarkToScreen(avg(23, 24));
        const hipOffX = AV.hipRest.x * AV.s;
        const hipOffY = AV.hipRest.y * AV.s;
        targetX = (hp.x - W / 2) - hipOffX;
        targetY = (H / 2 - hp.y) - hipOffY;
      } else {
        const shOffX = (AV.shRest ? AV.shRest.x : 0) * AV.s;
        const shOffY = (AV.shRest ? AV.shRest.y : (AV.hipRest.y + 0.44)) * AV.s;
        targetX = (sh.x - W / 2) - shOffX;
        targetY = (H / 2 - sh.y) - shOffY;
      }
      AV.pos.x += (targetX - AV.pos.x) * 0.14;
      AV.pos.y += (targetY - AV.pos.y) * 0.14;
      AV.rig.position.set(AV.pos.x, AV.pos.y, 0);
    }
    const switchBtn = document.getElementById("av-switch");
    if (switchBtn) {
      switchBtn.addEventListener("click", () => {
        const isFreyja = (AV.currentModel || "./assets/avatar2.glb").includes("avatar2");
        const nextModel = isFreyja ? "./assets/avatar.glb" : "./assets/avatar2.glb";
        AV.currentModel = nextModel;
        switchBtn.textContent = isFreyja ? "Model: Pandora" : "Model: Freyja";
        switchBtn.classList.toggle("active", !isFreyja);
        avatarStart(nextModel);
      });
    }
    document.getElementById("av-cal").addEventListener("click", avCalibrate);
    document.getElementById("av-reset").addEventListener("click", () => { AV.cal = null; });
    document.getElementById("av-bg").addEventListener("click", () => { AV.bg = (AV.bg + 1) % AV_BG.length; avCanvas.style.background = AV_BG[AV.bg]; });
    document.getElementById("av-skel").addEventListener("click", e => { state.showPose = !state.showPose; e.currentTarget.classList.toggle("active", state.showPose); });
    document.getElementById("av-load").addEventListener("click", () => document.getElementById("av-file").click());
    document.getElementById("av-file").addEventListener("change", e => { const f = e.target.files[0]; if (f) avatarStart(URL.createObjectURL(f)); });

    /* =========================================================================
       5. VISION ENGINE INITIALIZATION & HAIR SEGMENTATION
       ========================================================================= */
    const statusIndicator = document.getElementById("status-indicator");
    let gestureRecognizer = null;
    let faceLandmarker = null;
    let poseLandmarker = null;
    let imageSegmenter = null;
    let hairMaskData = null;
    let isSegmenting = false;
    let lastVideoTime = -1;
    let lastUploadedVideoTime = -1;
    let frameStep = 0;

    async function initVisionEngine() {
      try {
        const vision = await FilesetResolver.forVisionTasks(CONFIG.WASM_PATH);

        gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: { modelAssetPath: CONFIG.MODELS.GESTURE, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.52,
          minTrackingConfidence: 0.52
        });

        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: CONFIG.MODELS.FACE, delegate: "GPU" },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true
        });

        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: CONFIG.MODELS.POSE, delegate: "GPU" },
          runningMode: "VIDEO",
          numPoses: 1
        });

        imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: { modelAssetPath: CONFIG.MODELS.SEGMENTER, delegate: "GPU" },
          runningMode: "VIDEO",
          outputCategoryMask: false,
          outputConfidenceMasks: true
        }).catch(err => {
          console.warn("Hair segmenter fallback:", err);
          return null;
        });

        await startWebcam();
        await enumerateCameras();
      } catch (err) {
        console.warn("Camera fallback active:", err);
        state.hasCamera = false;
        statusIndicator.classList.add("standby");
      }
    }

    async function enumerateCameras() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        state.availableCameras = devices.filter(d => d.kind === "videoinput");
      } catch (e) {
        console.warn(e);
      }
    }

    /* Screen Wake Lock API (keeps screen active during AR studio sessions) */
    let wakeLockSentinel = null;

    async function requestWakeLock() {
      if ("wakeLock" in navigator && !wakeLockSentinel) {
        try {
          wakeLockSentinel = await navigator.wakeLock.request("screen");
          wakeLockSentinel.addEventListener("release", () => {
            wakeLockSentinel = null;
          });
        } catch (err) {
          wakeLockSentinel = null;
        }
      }
    }

    async function releaseWakeLock() {
      if (wakeLockSentinel) {
        try { await wakeLockSentinel.release(); } catch (e) {}
        wakeLockSentinel = null;
      }
    }

    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && state.hasCamera) {
        requestWakeLock();
      }
    });

    async function startWebcam(deviceId = null, facingMode = null) {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("getUserMedia unavailable");
      }

      if (video.srcObject) {
        try {
          const tracks = video.srcObject.getTracks();
          tracks.forEach(t => t.stop());
        } catch (e) {}
        video.srcObject = null;
        // Hardware camera release pause
        await new Promise(r => setTimeout(r, 60));
      }

      const prefFacing = facingMode || state.facingMode || "user";
      state.facingMode = prefFacing;

      let stream = null;
      try {
        const videoConstraints = deviceId
          ? { deviceId: { exact: deviceId } }
          : { width: { ideal: 1920, max: 1920 }, height: { ideal: 1080, max: 1080 }, facingMode: { ideal: prefFacing } };
        stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
      } catch (err1) {
        console.warn("Primary camera constraints failed, attempting fallback:", err1);
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: prefFacing },
            audio: false
          });
        } catch (err2) {
          console.warn("FacingMode fallback failed, attempting generic video:", err2);
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
      }

      video.srcObject = stream;
      await new Promise(resolve => {
        video.onloadedmetadata = () => {
          video.play().catch(() => {});
          resolve();
        };
      });

      state.hasCamera = true;
      statusIndicator.classList.remove("standby");
      fitViewport();

      // Inspect active track settings to identify rear/back camera
      const track = stream.getVideoTracks()[0];
      if (track && track.getSettings) {
        const settings = track.getSettings();
        if (settings.facingMode) {
          state.facingMode = settings.facingMode;
          if (settings.facingMode === "environment") {
            state.mirrorView = false;
            const bcf = document.getElementById("btn-camera-flip");
            if (bcf) bcf.classList.toggle("active", false);
          } else {
            state.mirrorView = true;
            const bcf = document.getElementById("btn-camera-flip");
            if (bcf) bcf.classList.toggle("active", true);
          }
        }
      }

      // Keep screen awake while camera is actively streaming
      requestWakeLock();
    }

    /* =========================================================================
       6. INTERFACE, SCENE SWITCHER & DRAWER CONTROLS
       ========================================================================= */
    const prismDock = document.getElementById("prism-dock");
    const makeupDock = document.getElementById("makeup-dock");
    const mysticDock = document.getElementById("mystic-dock");
    const statusBadge = document.getElementById("status-badge");

    // Scene Switching
    document.querySelectorAll(".scene-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        glCanvas.style.opacity = "1";
        document.querySelectorAll(".scene-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.scene = btn.dataset.scene;

        prismDock.style.display = state.scene === "PRISM" ? "flex" : "none";
        const musicDock = document.getElementById("music-dock");
        if (musicDock) musicDock.style.display = state.scene === "MUSIC" ? "flex" : "none";
        makeupDock.style.display = state.scene === "MAKEUP" ? "flex" : "none";
        mysticDock.style.display = state.scene === "MYSTIC" ? "flex" : "none";
        document.getElementById("game-dock").style.display = state.scene === "GAME" ? "flex" : "none";
        statusBadge.style.display = (state.scene === "GAME" || state.scene === "MYSTIC" || state.scene === "AVATAR" || state.scene === "MUSIC") ? "block" : "none";
        document.getElementById("avatar-dock").style.display = state.scene === "AVATAR" ? "flex" : "none";
        glCanvas.style.display = state.scene === "AVATAR" ? "none" : "block";
        avCanvas.style.display = state.scene === "AVATAR" ? "block" : "none";
        if (state.scene === "AVATAR" && !AV.ready) avatarStart();
        if (state.scene !== "AVATAR") {
          const avSkel = document.getElementById("av-skel");
          if (avSkel && avSkel.classList.contains("active")) {
            avSkel.classList.remove("active");
            state.showPose = false;
          }
        }

        if (state.scene === "GAME") {
          state.gameScore = 0;
          statusBadge.textContent = "PRANA SCORE: 0";
          resetGame();
        } else if (state.scene === "MYSTIC") {
          statusBadge.textContent = `JUTSU: ${state.spell}`;
        } else if (state.scene === "MUSIC") {
          statusBadge.textContent = `AIR MUSIC: ${state.instrument || "PIANO"}`;
        }
      });
    });

    // Desktop mouse wheel horizontal scrolling and smooth click-and-drag panning on small screen PC
    function enableMouseScroll(el) {
      if (!el) return;
      el.addEventListener("wheel", e => {
        if (e.deltaY !== 0) {
          e.preventDefault();
          el.scrollLeft += e.deltaY * 0.85;
        }
      }, { passive: false });

      // Only attach mouse drag on desktop pointer (not mobile touch screens)
      if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;

      let isDown = false;
      let startX = 0;
      let scrollLeft = 0;
      let hasDragged = false;

      el.addEventListener("mousedown", e => {
        if (e.button !== 0) return;
        isDown = true;
        hasDragged = false;
        startX = e.pageX - el.offsetLeft;
        scrollLeft = el.scrollLeft;
      });

      window.addEventListener("mouseup", () => {
        isDown = false;
      });

      el.addEventListener("mousemove", e => {
        if (!isDown) return;
        const x = e.pageX - el.offsetLeft;
        const walk = (x - startX) * 1.4;
        if (Math.abs(walk) > 6) hasDragged = true;
        el.scrollLeft = scrollLeft - walk;
      });

      el.addEventListener("click", e => {
        if (hasDragged) {
          e.stopImmediatePropagation();
          e.preventDefault();
          hasDragged = false;
        }
      }, true);
    }

    document.querySelectorAll(".dock-container, .scene-pill").forEach(enableMouseScroll);

    // Prism Shaders (Curated TouchDesigner-grade pipeline with true Spectral Prism first)
    const FILTERS = [
      "SPECTRAL_PRISM",
      "HOLO_FABRIC",
      "LIQUID_BUBBLE",
      "PRISM_PLEXUS",
      "PRISM_CRISSCROSS",
      "KALEIDOSCOPE",
      "ZOOM_LENS",
      "NEON_EDGE",
      "GAUSSIAN_SPLAT",
      "TOPOLOGICAL_MESH",
      "DEPTH_COLOR",
      "QUANTUM_GLITCH",
      "CYBER_XRAY",
      "HOLO_MATRIX",
      "THERMAL",
      "ANIME",
      "C41_NEGATIVE"
    ];
    function setFilterMode(f) {
      if (!FILTERS.includes(f)) return;
      state.filter = f;
      document.querySelectorAll("#prism-dock .filter-tab").forEach(tab => {
        tab.classList.toggle("active", tab.dataset.filter === f);
      });
      playAudioTone(440, "sine", 0.15);
    }

    document.querySelectorAll("#prism-dock .filter-tab").forEach(tab => {
      if (tab.id === "btn-prism-lock") return;
      tab.addEventListener("click", () => setFilterMode(tab.dataset.filter));
    });

    // Spatial Prism Quad Lock Button
    const btnPrismPrev = document.getElementById("btn-prism-prev");
    if (btnPrismPrev) {
      btnPrismPrev.addEventListener("click", () => {
        const curIdx = FILTERS.indexOf(state.filter);
        const prevIdx = (curIdx - 1 + FILTERS.length) % FILTERS.length;
        setFilterMode(FILTERS[prevIdx]);
        playAudioTone(580, "triangle", 0.12, 0.2);
        badge(`PRISM FILTER · ${FILTERS[prevIdx]}`);
      });
    }

    const btnPrismNext = document.getElementById("btn-prism-next");
    if (btnPrismNext) {
      btnPrismNext.addEventListener("click", () => {
        const curIdx = FILTERS.indexOf(state.filter);
        const nextIdx = (curIdx + 1) % FILTERS.length;
        setFilterMode(FILTERS[nextIdx]);
        playAudioTone(740, "triangle", 0.12, 0.2);
        badge(`PRISM FILTER · ${FILTERS[nextIdx]}`);
      });
    }

    const btnPrismLock = document.getElementById("btn-prism-lock");
    if (btnPrismLock) {
      btnPrismLock.addEventListener("click", () => {
        state.prismLocked = !state.prismLocked;
        if (state.prismLocked) {
          state.prismActive = true;
          state.prismOpacity = 1.0;
        }
        btnPrismLock.classList.toggle("active", state.prismLocked);
        btnPrismLock.textContent = state.prismLocked ? "Quad: Locked" : "Lock Quad";
        badge(state.prismLocked ? "SPATIAL PRISM · QUAD LOCKED IN PLACE" : "SPATIAL PRISM · TRACKING AIR PINCH");
        playAudioTone(state.prismLocked ? 720 : 480, "sine", 0.1);
      });
    }

    // Spatial Prism Mouse & Touch Screen Corner Dragging
    let draggingPrismCorner = null;

    function handlePrismPointerDown(e) {
      if (state.scene !== "PRISM") return;
      if (e.target.closest("button, .interactive, .dock-container, .scene-pill, .modal, .settings-drawer, .tour-card, input, select")) return;

      const rect = vecCanvas.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;

      const px = (e.clientX - rect.left) * (vecCanvas.width / rect.width);
      const py = (e.clientY - rect.top) * (vecCanvas.height / rect.height);
      const cw = vecCanvas.width, ch = vecCanvas.height;

      // Only allow corner handle dragging if prism quad is already active
      if (!state.prismActive) return;

      const cornerCoords = {
        tl: { x: state.corners.tl.x * cw, y: state.corners.tl.y * ch },
        tr: { x: state.corners.tr.x * cw, y: state.corners.tr.y * ch },
        br: { x: state.corners.br.x * cw, y: state.corners.br.y * ch },
        bl: { x: state.corners.bl.x * cw, y: state.corners.bl.y * ch }
      };

      let closestCorner = null;
      let minDist = 55; // 55px touch target radius
      for (const [k, pt] of Object.entries(cornerCoords)) {
        const d = Math.hypot(px - pt.x, py - pt.y);
        if (d < minDist) {
          minDist = d;
          closestCorner = k;
        }
      }

      if (closestCorner) {
        draggingPrismCorner = closestCorner;
      }
    }

    function handlePrismPointerMove(e) {
      if (state.scene !== "PRISM" || !draggingPrismCorner) return;
      const rect = vecCanvas.getBoundingClientRect();
      const px = (e.clientX - rect.left) * (vecCanvas.width / rect.width);
      const py = (e.clientY - rect.top) * (vecCanvas.height / rect.height);
      const normX = Math.max(0.0, Math.min(1.0, px / vecCanvas.width));
      const normY = Math.max(0.0, Math.min(1.0, py / vecCanvas.height));
      state.targetCorners[draggingPrismCorner].x = normX;
      state.targetCorners[draggingPrismCorner].y = normY;
    }

    function handlePrismPointerUp() {
      draggingPrismCorner = null;
    }

    window.addEventListener("pointerdown", handlePrismPointerDown, { passive: true });
    window.addEventListener("pointermove", handlePrismPointerMove, { passive: true });
    window.addEventListener("pointerup", handlePrismPointerUp, { passive: true });
    window.addEventListener("pointercancel", handlePrismPointerUp, { passive: true });

    // Mystic Live Feed / Galaxy Background Toggle
    const btnMysticFeed = document.getElementById("btn-mystic-feed");
    if (btnMysticFeed) {
      btnMysticFeed.addEventListener("click", () => {
        state.mysticGalaxy = !state.mysticGalaxy;
        btnMysticFeed.textContent = state.mysticGalaxy ? "Feed: Galaxy" : "Feed: Camera";
        btnMysticFeed.classList.toggle("active", state.mysticGalaxy);
        const hideCamFeed = (state.scene === "MYSTIC" && state.mysticGalaxy) || (state.scene === "GAME" && state.gameGalaxy);
        glCanvas.style.opacity = hideCamFeed ? "0" : "1";
      });
    }

    // Mystic Spells
    document.querySelectorAll("#mystic-dock .filter-tab").forEach(tab => {
        if (tab.id === "btn-mystic-feed") return;
      tab.addEventListener("click", () => {
        if (tab.dataset.spell === "TEXT" && state.spell === "TEXT") {
          const v = prompt("Title | Subtitle | Caption   (separate lines with |)", txt.lines.join(" | "));
          if (v && v.trim()) txt.lines = v.split("|").map(q => q.trim().toUpperCase().slice(0, 28)).filter(Boolean).slice(0, 3);
          return;
        }
        document.querySelectorAll("#mystic-dock .filter-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        state.spell = tab.dataset.spell;
        statusBadge.textContent = `JUTSU: ${state.spell}`;
        playAudioTone(520, "triangle", 0.2);
      });
    });

    // Prana Catch modes & Educational Curriculum
    const eduModal = document.getElementById("edu-modal");
    document.querySelectorAll("#game-dock .filter-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        if (tab.id === "btn-custom-edu") {
          if (eduModal) eduModal.style.display = "flex";
          return;
        }
        document.querySelectorAll("#game-dock .filter-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        state.gmode = tab.dataset.gmode || "ARCADE";
        resetGame();
        playAudioTone(520, "triangle", 0.2);
      });
    });

    // Educational modal handlers
    document.getElementById("btn-edu-close")?.addEventListener("click", () => {
      if (eduModal) eduModal.style.display = "none";
    });
    document.querySelectorAll(".btn-pack-select").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".btn-pack-select").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const pack = btn.dataset.pack;
        state.gmode = pack;
        document.querySelectorAll("#game-dock .filter-tab").forEach(t => {
          t.classList.toggle("active", t.dataset.gmode === pack);
        });
        resetGame();
        playAudioTone(520, "triangle", 0.2);
      });
    });
    document.getElementById("btn-edu-load-custom")?.addEventListener("click", () => {
      const input = document.getElementById("edu-custom-input");
      const txt = (input && input.value || "").trim();
      if (!txt) return;
      const parsed = [];
      txt.split("\n").forEach(line => {
        const parts = line.split(",").map(s => s.trim());
        if (parts.length >= 2) {
          const q = parts[0];
          const a = parts[1];
          const w = parts[2] || "";
          parsed.push([q, a, w]);
        }
      });
      if (parsed.length > 0) {
        QBANK.CUSTOM = parsed;
        state.gmode = "CUSTOM";
        document.querySelectorAll("#game-dock .filter-tab").forEach(t => t.classList.remove("active"));
        const customTab = document.getElementById("btn-custom-edu");
        if (customTab) { customTab.classList.add("active"); customTab.textContent = `Custom (${parsed.length})`; }
        if (eduModal) eduModal.style.display = "none";
        resetGame();
        badge(`CUSTOM PACK LOADED · ${parsed.length} QUESTIONS`);
        playAudioTone(660, "triangle", 0.25);
      }
    });
    document.getElementById("btn-edu-reset-default")?.addEventListener("click", () => {
      const input = document.getElementById("edu-custom-input");
      if (input) input.value = "";
      QBANK.CUSTOM = [];
      const customTab = document.getElementById("btn-custom-edu");
      if (customTab) customTab.textContent = "Custom Quiz";
      badge("CUSTOM PACK CLEARED");
    });

    // Hair Swatches
    document.querySelectorAll("[data-hair]").forEach(sw => {
      sw.addEventListener("click", () => {
        document.querySelectorAll("[data-hair]").forEach(s => s.classList.remove("active"));
        sw.classList.add("active");
        state.hairShade = sw.dataset.hair;
      });
    });

    // Lipstick Swatches
    document.querySelectorAll("[data-lip]").forEach(sw => {
      sw.addEventListener("click", () => {
        document.querySelectorAll("[data-lip]").forEach(s => s.classList.remove("active"));
        sw.classList.add("active");
        state.lipColor = sw.dataset.lip;
      });
    });

    // Custom Color Pickers and Blush Swatches
    const pickerLip = document.getElementById("picker-lip-color");
    if (pickerLip) {
      pickerLip.addEventListener("input", e => {
        document.querySelectorAll("[data-lip]").forEach(s => s.classList.remove("active"));
        state.lipColor = e.target.value;
      });
    }

    const pickerHair = document.getElementById("picker-hair-color");
    if (pickerHair) {
      pickerHair.addEventListener("input", e => {
        document.querySelectorAll("[data-hair]").forEach(s => s.classList.remove("active"));
        const hex = e.target.value;
        HAIR_PALETTES["custom"] = [hex, hex];
        state.hairShade = "custom";
      });
    }

    document.querySelectorAll("[data-blush]").forEach(sw => {
      sw.addEventListener("click", () => {
        document.querySelectorAll("[data-blush]").forEach(s => s.classList.remove("active"));
        sw.classList.add("active");
        state.blushColor = sw.dataset.blush || sw.getAttribute("data-blush");
        playAudioTone(520, "sine", 0.08, 0.15);
      });
    });

    const pickerBlush = document.getElementById("picker-blush-color");
    if (pickerBlush) {
      pickerBlush.addEventListener("input", e => {
        document.querySelectorAll("[data-blush]").forEach(s => s.classList.remove("active"));
        state.blushColor = e.target.value;
      });
    }

    // Sliders
    function setupSlider(id, valId, key, options = {}) {
      const slider = document.getElementById(id);
      const valDisplay = document.getElementById(valId);
      if (!slider || !valDisplay) return;

      const unit = options.unit !== undefined ? options.unit : "%";
      const divisor = options.divisor !== undefined ? options.divisor : 100.0;

      const update = () => {
        const val = parseFloat(slider.value);
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
        slider.style.background = `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${pct}%, #232733 ${pct}%, #232733 100%)`;
        valDisplay.textContent = unit ? `${val}${unit}` : `${val}`;
        state[key] = divisor === 1 ? val : val / divisor;
      };

      slider.addEventListener("input", update);
      update();
    }

    setupSlider("slider-density", "val-density", "density");
    setupSlider("slider-radius", "val-radius", "splatRadius");
    setupSlider("slider-edge", "val-edge", "edgeGain");
    setupSlider("slider-vibrance", "val-vibrance", "vibrance");
    setupSlider("slider-skin", "val-skin", "skinSmooth");
    setupSlider("slider-zoom", "val-zoom", "zoom");
    setupSlider("slider-lip-opacity", "val-lip-opacity", "lipOpacity");
    setupSlider("slider-blush-opacity", "val-blush-opacity", "blushOpacity");
    setupSlider("slider-plexus-radius", "val-plexus-radius", "plexusRadius", { unit: "px", divisor: 1 });
    setupSlider("slider-flock-count", "val-flock-count", "flockCount", { unit: "", divisor: 1 });
    setupSlider("slider-flock-speed", "val-flock-speed", "flockSpeed");

    // Camera & Audio Actions
    const btnCamFlip = document.getElementById("btn-camera-flip");
    btnCamFlip.addEventListener("click", () => {
      state.mirrorView = !state.mirrorView;
      btnCamFlip.classList.toggle("active", state.mirrorView);
      btnCamFlip.title = state.mirrorView ? "Mirror View: ON (Active)" : "Mirror View: OFF (Natural)";
      badge(state.mirrorView ? "MIRROR VIEW · ON (FLIPPED)" : "MIRROR VIEW · OFF (NATURAL)");
      playAudioTone(state.mirrorView ? 660 : 440, "sine", 0.08);
    });

    const btnAudio = document.getElementById("btn-audio-toggle");
    btnAudio.addEventListener("click", () => {
      state.soundEnabled = !state.soundEnabled;
      btnAudio.classList.toggle("active", state.soundEnabled);
    });

    const btnCamSwap = document.getElementById("btn-camera-swap");
    async function switchCamera() {
      if (state.availableCameras.length <= 1) await enumerateCameras();

      let nextIsEnv = false;
      try {
        if (state.availableCameras.length > 1) {
          state.currentCameraIndex = (state.currentCameraIndex + 1) % state.availableCameras.length;
          const target = state.availableCameras[state.currentCameraIndex];
          const label = (target.label || "").toLowerCase();
          nextIsEnv = label.includes("back") || label.includes("rear") || label.includes("environment");
          try {
            await startWebcam(target.deviceId, nextIsEnv ? "environment" : "user");
          } catch (e) {
            console.warn("Device ID switch failed, falling back to facingMode toggle:", e);
            state.facingMode = state.facingMode === "user" ? "environment" : "user";
            nextIsEnv = state.facingMode === "environment";
            try {
              await startWebcam(null, state.facingMode);
            } catch (e2) {
              console.warn("Fallback camera start failed:", e2);
            }
          }
        } else {
          // Toggle user <-> environment
          state.facingMode = (state.facingMode === "environment") ? "user" : "environment";
          nextIsEnv = state.facingMode === "environment";
          try {
            await startWebcam(null, state.facingMode);
          } catch (e) {
            console.warn("FacingMode switch failed:", e);
          }
        }

        // Automatically mirror front camera and un-mirror back camera
        state.mirrorView = !nextIsEnv;
        if (btnCamFlip) btnCamFlip.classList.toggle("active", state.mirrorView);

        // Re-enumerate cameras in case labels were unlocked
        await enumerateCameras();

        // Show brief status feedback
        const sb = document.getElementById("status-badge");
        if (sb) {
          const prevText = sb.textContent;
          sb.textContent = nextIsEnv ? "CAMERA: REAR / ENVIRONMENT (MIRROR OFF)" : "CAMERA: FRONT / USER (MIRROR ON)";
          sb.style.display = "block";
          setTimeout(() => { sb.textContent = prevText; }, 2000);
        }
      } catch (err) {
        console.warn("Camera switch error handled safely:", err);
        const sb = document.getElementById("status-badge");
        if (sb) {
          const prevText = sb.textContent;
          sb.textContent = "CAMERA UNAVAILABLE / OCCUPIED";
          sb.style.display = "block";
          setTimeout(() => { sb.textContent = prevText; }, 2500);
        }
      }
    }
    btnCamSwap.addEventListener("click", switchCamera);

    // Drawer Toggles
    const settingsDrawer = document.getElementById("settings-drawer");
    document.getElementById("btn-open-settings").addEventListener("click", () => settingsDrawer.classList.add("open"));
    document.getElementById("btn-close-settings").addEventListener("click", () => settingsDrawer.classList.remove("open"));

    const toggleLock = document.getElementById("toggle-prism-lock");
    toggleLock.addEventListener("click", () => {
      state.prismLocked = !state.prismLocked;
      toggleLock.textContent = state.prismLocked ? "ON" : "OFF";
      toggleLock.classList.toggle("active", state.prismLocked);
      if (state.prismLocked) state.prismActive = true;
    });

    function setupToggle(id, key) {
      const btn = document.getElementById(id);
      btn.addEventListener("click", () => {
        state[key] = !state[key];
        btn.textContent = state[key] ? "ON" : "OFF";
        btn.classList.toggle("active", state[key]);
      });
    }

    setupToggle("toggle-hands", "showHands");
    setupToggle("toggle-pose", "showPose");
    setupToggle("toggle-face", "showFaceMesh");

    // Onboarding Modal
    const onboardModal = document.getElementById("onboard-modal");
    document.getElementById("btn-enter").addEventListener("click", () => onboardModal.style.display = "none");
    document.getElementById("btn-quick-settings").addEventListener("click", () => {
      onboardModal.style.display = "none";
      settingsDrawer.classList.add("open");
    });
    document.getElementById("btn-brand").addEventListener("click", () => onboardModal.style.display = "flex");
    document.getElementById("btn-show-instructions").addEventListener("click", () => {
      settingsDrawer.classList.remove("open");
      onboardModal.style.display = "flex";
    });

    /* =========================================================================
       7. MAIN COMPOSITOR LOOP & UNCLAMPED UPWARD EXPANSION
       ========================================================================= */
    let lastFrameTime = performance.now();

    function renderLoop(timeMs) {
      requestAnimationFrame(renderLoop);
      const dt = Math.min((timeMs - lastFrameTime) / 1000.0, 0.1);
      lastFrameTime = timeMs;

      let activeImageSource = null;
      let handsDetected = false;

      if (state.hasCamera && video.readyState >= 2) {
        activeImageSource = video;

        if (gestureRecognizer && video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          frameStep++;

          // Scene-aware tracker throttling for smooth performance:
          // Scene-aware tracker throttling for smooth performance:
          // In Avatar mode, run pose every 2 frames, face every 3 frames, and hands every 3 frames.
          const isAvatarScene = state.scene === "AVATAR";

          // Hands / Gestures Tracking
          if (gestureRecognizer && (!isAvatarScene || frameStep % 3 === 0)) {
            const handRes = gestureRecognizer.recognizeForVideo(video, timeMs);
            state.handData = [];

            if (handRes.landmarks && handRes.landmarks.length > 0) {
              handsDetected = true;
              state.lastHandsTime = timeMs;
              let leftHand = null;
              let rightHand = null;

              handRes.landmarks.forEach(lms => {
                const wrist = lms[0];
                const middleMcp = lms[9];
                const thumbTip = lms[4];
                const indexTip = lms[8];

                const handScale = Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y) || 0.15;
                const pinchDist = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y);
                const isPinching = (pinchDist / handScale) < 0.45;

                const sThumb = landmarkToScreen(thumbTip);
                const sIndex = landmarkToScreen(indexTip);
                const sMid = {
                  normX: (sThumb.normX + sIndex.normX) * 0.5,
                  normY: (sThumb.normY + sIndex.normY) * 0.5
                };

                const entry = { landmarks: lms, isPinching, sThumb, sIndex, sMid };
                state.handData.push(entry);
              });

              if (!isAvatarScene) {
                // Smooth, full-screen horizontal boundary extrapolation (effortlessly reaches 0.0 and 1.0)
                const snapEdgeX = (normX, isLeft) => {
                  if (isLeft) {
                    if (normX <= 0.24) {
                      const u = Math.max(0.0, (normX - 0.05) / (0.24 - 0.05));
                      return u * 0.24;
                    }
                  } else {
                    if (normX >= 0.76) {
                      const u = Math.min(1.0, (normX - 0.76) / (0.95 - 0.76));
                      return 0.76 + u * 0.24;
                    }
                  }
                  return normX;
                };

                if (state.handData.length >= 2) {
                  const sorted = [...state.handData].sort((a, b) => a.sMid.normX - b.sMid.normX);
                  leftHand = sorted[0]; rightHand = sorted[sorted.length - 1];
                  state.lastLeftHand = { sIndex: { ...leftHand.sIndex }, sThumb: { ...leftHand.sThumb }, isPinching: leftHand.isPinching };
                  state.lastRightHand = { sIndex: { ...rightHand.sIndex }, sThumb: { ...rightHand.sThumb }, isPinching: rightHand.isPinching };
                } else if (state.handData.length === 1) {
                  // Single hand visible (e.g. mobile phone in one hand, pinching with other):
                  // Anchor out-of-frame side to screen edge
                  const single = state.handData[0];
                  if (single.sMid.normX < 0.5) {
                    leftHand = single;
                    rightHand = {
                      sIndex: { normX: 1.0, normY: single.sIndex.normY },
                      sThumb: { normX: 1.0, normY: single.sThumb.normY },
                      isPinching: single.isPinching
                    };
                  } else {
                    rightHand = single;
                    leftHand = {
                      sIndex: { normX: 0.0, normY: single.sIndex.normY },
                      sThumb: { normX: 0.0, normY: single.sThumb.normY },
                      isPinching: single.isPinching
                    };
                  }
                  state.lastLeftHand = { sIndex: { ...leftHand.sIndex }, sThumb: { ...leftHand.sThumb }, isPinching: leftHand.isPinching };
                  state.lastRightHand = { sIndex: { ...rightHand.sIndex }, sThumb: { ...rightHand.sThumb }, isPinching: rightHand.isPinching };
                }

                const dualPinch = leftHand && rightHand && leftHand.isPinching && rightHand.isPinching;
                if (dualPinch) state.prismActive = true;

                if ((state.prismActive || state.prismLocked) && leftHand && rightHand) {
                  state.targetCorners.tl.x = snapEdgeX(leftHand.sIndex.normX, true);
                  state.targetCorners.tl.y = leftHand.sIndex.normY;
                  state.targetCorners.bl.x = snapEdgeX(leftHand.sThumb.normX, true);
                  state.targetCorners.bl.y = leftHand.sThumb.normY;
                  state.targetCorners.tr.x = snapEdgeX(rightHand.sIndex.normX, false);
                  state.targetCorners.tr.y = rightHand.sIndex.normY;
                  state.targetCorners.br.x = snapEdgeX(rightHand.sThumb.normX, false);
                  state.targetCorners.br.y = rightHand.sThumb.normY;
                }

                // Pinch shutter removed: capture strictly via on-screen buttons and Space/C hotkeys
              }

            } else if (!state.prismLocked && timeMs - state.lastHandsTime > 550) {
              state.prismActive = false; state.shutterArmed = false; state.pinchSince = 0; state.lastLeftHand = null; state.lastRightHand = null;
            }
          }

          // Face Mesh & Wink Detection (runs across all modes for realistic head/face tracking)
          const needFace = state.scene === "MAKEUP" || state.scene === "GAME" || isAvatarScene || state.showFaceMesh;
          const faceInterval = needFace ? (isAvatarScene ? 3 : 2) : 6;
          if (faceLandmarker && frameStep % faceInterval === 1) {
            const faceRes = faceLandmarker.detectForVideo(video, timeMs);
            if (faceRes.faceLandmarks && faceRes.faceLandmarks.length > 0) {
              state.faceLandmarks = faceRes.faceLandmarks[0];
            } else {
              state.faceLandmarks = null;
            }

            if (faceRes.faceBlendshapes && faceRes.faceBlendshapes.length > 0) {
              const cats = faceRes.faceBlendshapes[0].categories;
              const blinkL = cats.find(c => c.categoryName === "eyeBlinkLeft")?.score || 0;
              const blinkR = cats.find(c => c.categoryName === "eyeBlinkRight")?.score || 0;
              const jaw = cats.find(c => c.categoryName === "jawOpen")?.score || 0;
              state.jawOpen = jaw;

              if (state.scene === "PRISM" && Math.abs(blinkL - blinkR) > 0.45 && Math.min(blinkL, blinkR) < 0.3 && (timeMs - state.lastWinkTime > 900)) {
                state.lastWinkTime = timeMs;
                const nextIdx = (FILTERS.indexOf(state.filter) + 1) % FILTERS.length;
                setFilterMode(FILTERS[nextIdx]);
                playAudioTone(720, "triangle", 0.12, 0.2);
                badge(`PRISM FILTER · ${FILTERS[nextIdx]}`);
              }

              // Head Turn Left / Right gesture navigation for Spatial Prism
              if (state.scene === "PRISM" && state.faceLandmarks) {
                const fl = state.faceLandmarks;
                const nose = fl[1], leftCh = fl[234], rightCh = fl[454];
                if (nose && leftCh && rightCh) {
                  const faceW = Math.abs(rightCh.x - leftCh.x) || 0.2;
                  const rawYaw = (nose.x - (leftCh.x + rightCh.x) * 0.5) / faceW;
                  const yaw = state.mirrorView ? -rawYaw : rawYaw;

                  if (Math.abs(yaw) < 0.08) {
                    state.headTurnArmed = true;
                  } else if (state.headTurnArmed && (timeMs - (state.lastHeadTurnTime || 0) > 750)) {
                    if (yaw > 0.20) {
                      state.lastHeadTurnTime = timeMs;
                      state.headTurnArmed = false;
                      const curIdx = FILTERS.indexOf(state.filter);
                      const nextIdx = (curIdx + 1) % FILTERS.length;
                      setFilterMode(FILTERS[nextIdx]);
                      playAudioTone(760, "triangle", 0.12, 0.25);
                      badge(`HEAD TURN RIGHT · ${FILTERS[nextIdx]}`);
                    } else if (yaw < -0.20) {
                      state.lastHeadTurnTime = timeMs;
                      state.headTurnArmed = false;
                      const curIdx = FILTERS.indexOf(state.filter);
                      const prevIdx = (curIdx - 1 + FILTERS.length) % FILTERS.length;
                      setFilterMode(FILTERS[prevIdx]);
                      playAudioTone(560, "triangle", 0.12, 0.25);
                      badge(`HEAD TURN LEFT · ${FILTERS[prevIdx]}`);
                    }
                  }
                }
              }

              // Horizontal Hand Swipe gesture navigation for Spatial Prism
              if (state.scene === "PRISM" && state.handData.length > 0) {
                const curHandX = state.handData[0].sMid.normX;
                if (prevHandX !== null && prevHandTime > 0 && (timeMs - prevHandTime) < 250) {
                  const dt = Math.max(0.016, (timeMs - prevHandTime) / 1000);
                  const dx = curHandX - prevHandX;
                  const vx = dx / dt;

                  if ((timeMs - lastHandSwipeTime > 650) && Math.abs(vx) > 1.4 && Math.abs(dx) > 0.07) {
                    lastHandSwipeTime = timeMs;
                    const curIdx = FILTERS.indexOf(state.filter);
                    if (vx > 0) {
                      // Swipe Right -> Next Filter
                      const nextIdx = (curIdx + 1) % FILTERS.length;
                      setFilterMode(FILTERS[nextIdx]);
                      playAudioTone(740, "triangle", 0.12, 0.25);
                      badge(`SWIPE NEXT · ${FILTERS[nextIdx]}`);
                    } else {
                      // Swipe Left -> Prev Filter
                      const prevIdx = (curIdx - 1 + FILTERS.length) % FILTERS.length;
                      setFilterMode(FILTERS[prevIdx]);
                      playAudioTone(580, "triangle", 0.12, 0.25);
                      badge(`SWIPE PREV · ${FILTERS[prevIdx]}`);
                    }
                  }
                }
                prevHandX = curHandX;
                prevHandTime = timeMs;
              } else if (state.handData.length === 0) {
                prevHandX = null;
              }
            }
          }

          // Pose Landmarks — run only in Avatar mode or when skeleton overlay is toggled on (eliminates background stutter)
          const needPose = isAvatarScene || state.showPose || (state.scene === "MYSTIC" && state.spell === "PLEXUS");
          const poseInterval = (isAvatarScene || (state.scene === "MYSTIC" && state.spell === "PLEXUS")) ? 2 : 4;
          if (poseLandmarker && needPose && frameStep % poseInterval === 0) {
            const poseRes = poseLandmarker.detectForVideo(video, timeMs);
            state.poseLandmarks = (poseRes.landmarks && poseRes.landmarks.length > 0) ? poseRes.landmarks[0] : null;
            state.poseWorld = (poseRes.worldLandmarks && poseRes.worldLandmarks.length > 0) ? poseRes.worldLandmarks[0] : null;
          } else if (!needPose) {
            state.poseLandmarks = null;
            state.poseWorld = null;
          }

          // AI Hair Segmentation (Lustra Asynchronous Multiclass)
          if ((state.scene === "MAKEUP" || (state.scene === "PRISM" && state.filter === "DEPTH_COLOR")) && imageSegmenter && !isSegmenting && frameStep % 3 === 0) {
            isSegmenting = true;
            try {
              imageSegmenter.segmentForVideo(video, timeMs, result => {
                try {
                  const cm = result.confidenceMasks, mh_ = cm && cm[1], ms_ = cm && cm[3], mb_ = cm && cm[0]; // 1 hair, 3 face-skin, 0 background
                  if (mh_) {
                    const A = mh_.getAsFloat32Array(), B = ms_ ? ms_.getAsFloat32Array() : null, Z = mb_ ? mb_.getAsFloat32Array() : null;
                    if (maskCanvas.width !== mh_.width) {
                      maskCanvas.width = mh_.width; maskCanvas.height = mh_.height;
                      prevHair = new Float32Array(A.length); prevSkin = new Float32Array(A.length); prevPerson = new Float32Array(A.length);
                    }
                    const id = maskCtx.createImageData(mh_.width, mh_.height);
                    for (let i = 0; i < A.length; i++) {
                      prevHair[i] += (A[i] - prevHair[i]) * 0.55; // temporal smoothing = no flicker
                      if (B) prevSkin[i] += (B[i] - prevSkin[i]) * 0.55;
                      if (Z) prevPerson[i] += ((1 - Z[i]) - prevPerson[i]) * 0.55;
                      const p = i * 4;
                      id.data[p] = prevHair[i] * 255; id.data[p + 1] = prevSkin[i] * 255; id.data[p + 2] = prevPerson[i] * 255; id.data[p + 3] = 255;
                    }
                    maskCtx.putImageData(id, 0, 0);
                    state.maskDirty = true;
                  }
                } finally {
                  isSegmenting = false;
                }
              });
            } catch (err) {
              isSegmenting = false;
            }
          }
        }
      }

      // Smooth Transitions: Prism Fade-in and Fade-out
      const targetAlpha = (state.prismActive && (timeMs - state.lastHandsTime < 450 || state.prismLocked)) ? 1.0 : 0.0;
      state.prismOpacity += (targetAlpha - state.prismOpacity) * Math.min(1.0, dt * 14.0);

      // Snap corners to the fingers on engage (no swoosh from the old position)
      if (state.prismActive && !state.prevActive) {
        for (const k of ["tl", "tr", "br", "bl"]) { state.corners[k].x = state.targetCorners[k].x; state.corners[k].y = state.targetCorners[k].y; }
      }
      state.prevActive = state.prismActive;

      // Smooth Corner Interpolation
      const lerpSpeed = Math.min(1.0, dt * 18.0);
      state.corners.tl.x += (state.targetCorners.tl.x - state.corners.tl.x) * lerpSpeed;
      state.corners.tl.y += (state.targetCorners.tl.y - state.corners.tl.y) * lerpSpeed;
      state.corners.tr.x += (state.targetCorners.tr.x - state.corners.tr.x) * lerpSpeed;
      state.corners.tr.y += (state.targetCorners.tr.y - state.corners.tr.y) * lerpSpeed;
      state.corners.br.x += (state.targetCorners.br.x - state.corners.br.x) * lerpSpeed;
      state.corners.br.y += (state.targetCorners.br.y - state.corners.br.y) * lerpSpeed;
      state.corners.bl.x += (state.targetCorners.bl.x - state.corners.bl.x) * lerpSpeed;
      state.corners.bl.y += (state.targetCorners.bl.y - state.corners.bl.y) * lerpSpeed;

      // WebGL Shader Render
      if (activeImageSource) {
        gl.useProgram(glProgram);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, videoTexture);
        if (video.currentTime !== lastUploadedVideoTime) {
          lastUploadedVideoTime = video.currentTime;
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, activeImageSource);
        }
        gl.uniform1i(uVideoLoc, 0);

        gl.uniform2f(uResLoc, glCanvas.width, glCanvas.height);
        gl.uniform1i(uMirrorLoc, state.mirrorView ? 1 : 0);
        // Preserve prism during gesture-capture so the shader effect is included in the screenshot
        gl.uniform1f(uPrismAlphaLoc, state.scene === "PRISM" ? state.prismOpacity : 0.0);

        const filterIndex = FILTERS.indexOf(state.filter);
        gl.uniform1i(uFilterLoc, filterIndex >= 0 ? filterIndex : 0);

        gl.uniform1f(uDensityLoc, state.density);
        gl.uniform1f(uSplatRadiusLoc, state.splatRadius);
        gl.uniform1f(uEdgeGainLoc, state.edgeGain);
        gl.uniform1f(uTime, timeMs / 1000);
        gl.uniform1f(uZoomL, state.zoom);
        const hp0 = state.handData[0];
        gl.uniform2f(uPointer, hp0 ? hp0.sIndex.normX : 0.5, hp0 ? hp0.sIndex.normY : 0.5);
        gl.uniform1f(uPointerOn, hp0 ? 1 : 0);
        const mkOn = state.scene === "MAKEUP";
        gl.uniform1i(uMakeupL, mkOn ? 1 : 0);
        if (state.scene === "PRISM" && state.filter === "DEPTH_COLOR") {
          gl.activeTexture(gl.TEXTURE1);
          gl.bindTexture(gl.TEXTURE_2D, maskTex);
          if (state.maskDirty) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, maskCanvas); state.maskDirty = false; }
          gl.uniform1i(uMaskL, 1);
          gl.activeTexture(gl.TEXTURE0);
        }
        if (mkOn) {
          const pal = HAIR_PALETTES[state.hairShade];
          const cs = (pal || ["#000000"]).map(hexRgb), C = cs[cs.length - 1];
          const B = cs.length === 3 ? cs[1] : cs[0].map((v, i) => (v + C[i]) / 2);
          gl.uniform3f(uHairAL, ...cs[0]); gl.uniform3f(uHairBL, ...B); gl.uniform3f(uHairCL, ...C);
          gl.uniform1f(uHairOnL, pal ? 1 : 0);
          gl.uniform1f(uVibL, state.vibrance);
          gl.uniform1f(uSmoothL, state.skinSmooth);
          gl.activeTexture(gl.TEXTURE1);
          gl.bindTexture(gl.TEXTURE_2D, maskTex);
          if (state.maskDirty) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, maskCanvas); state.maskDirty = false; }
          gl.uniform1i(uMaskL, 1);
          gl.activeTexture(gl.TEXTURE2);
          gl.bindTexture(gl.TEXTURE_2D, mkTex);
          if (mkCanvas.width > 1) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, mkCanvas);
          gl.uniform1i(uMkL, 2);
          gl.activeTexture(gl.TEXTURE0);
        }

        // Cyclic convex order: TL -> TR -> BR -> BL
        gl.uniform2f(uV0Loc, state.corners.tl.x, state.corners.tl.y);
        gl.uniform2f(uV1Loc, state.corners.tr.x, state.corners.tr.y);
        gl.uniform2f(uV2Loc, state.corners.br.x, state.corners.br.y);
        gl.uniform2f(uV3Loc, state.corners.bl.x, state.corners.bl.y);

        gl.bindVertexArray(quadVao);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }

      if (state.scene === "AVATAR") avatarFrame();
      render2DOverlays();
      if (state.capture) { const captureSource = state.capture; state.capture = false; snapshot(captureSource); }
    }

    /* =========================================================================
       8. STUDIO TOOLS: SNAPSHOT, FULLSCREEN, PWA INSTALL
       ========================================================================= */
    function snapshot(captureSource) {
      const c = document.createElement("canvas");
      c.width = glCanvas.width; c.height = glCanvas.height;
      const x = c.getContext("2d");

      // Composite the right source canvas
      if (state.scene === "AVATAR") {
        const avBgColors = ["#0a0d1a", "#070a18", "#000000", "#004d1a"];
        x.fillStyle = avBgColors[AV.bg] || "#0a0d1a";
        x.fillRect(0, 0, c.width, c.height);
        x.drawImage(avCanvas, 0, 0, c.width, c.height);
      } else {
        x.drawImage(glCanvas, 0, 0);
      }

      // Always overlay 2D vector canvas to retain spatial prism reticle, mystic spells, and game visuals
      x.drawImage(vecCanvas, 0, 0);

      // ---- Stylish, discreet two-row watermark (bottom-right) ----
      const cw = c.width, ch = c.height;
      const fs1 = Math.max(9, Math.round(cw / 120));   // Row 1 font size (clean proportion)
      const fs2 = Math.max(7, Math.round(cw / 175));   // Row 2 font size (small, subtle)
      const padX = Math.round(fs1 * 0.85);
      const padY = Math.round(fs1 * 0.45);
      const gap = Math.round(fs1 * 0.22);

      x.font = `600 ${fs1}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
      const w1 = x.measureText("Divya Drusti").width;
      x.font = `500 ${fs2}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
      const w2 = x.measureText("Razel Tech").width;
      const maxW = Math.max(w1, w2);
      const pillW = maxW + padX * 2;
      const pillH = fs1 + gap + fs2 + padY * 2;

      // Discreet frosted dark pill
      const inset = Math.max(12, Math.round(cw * 0.016));
      const px = cw - pillW - inset;
      const py = ch - pillH - inset;

      x.save();
      x.fillStyle = "rgba(6, 7, 10, 0.65)";
      x.strokeStyle = "rgba(255, 255, 255, 0.09)";
      x.lineWidth = 1.0;
      if (x.roundRect) {
        x.beginPath();
        x.roundRect(px, py, pillW, pillH, 5);
        x.fill();
        x.stroke();
      } else {
        x.fillRect(px, py, pillW, pillH);
        x.strokeRect(px, py, pillW, pillH);
      }

      // Row 1: Divya Drusti
      x.fillStyle = "rgba(243, 245, 248, 0.94)";
      x.font = `600 ${fs1}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
      x.textAlign = "left";
      x.textBaseline = "top";
      x.fillText("Divya Drusti", px + padX, py + padY);

      // Row 2: Razel Tech
      x.fillStyle = "rgba(96, 165, 250, 0.82)";
      x.font = `500 ${fs2}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
      x.fillText("Razel Tech", px + padX, py + padY + fs1 + gap);
      x.restore();

      c.toBlob(b => {
        state.lastSnapshotBlob = b;
        if (state.lastSnapshotUrl) {
          URL.revokeObjectURL(state.lastSnapshotUrl);
        }
        const url = URL.createObjectURL(b);
        state.lastSnapshotUrl = url;

        const previewModal = document.getElementById("snapshot-modal");
        const previewImg = document.getElementById("snapshot-preview-img");
        if (previewModal && previewImg) {
          previewImg.src = url;
          previewModal.style.display = "flex";
        }
      }, "image/png");
      playAudioTone(1200, "square", 0.05, 0.08);
      state.shotFlash = 1;
      const sb = document.getElementById("status-badge"), prevDisp = sb.style.display;
      sb.style.display = "block"; sb.textContent = "EFFECT CAPTURED · PREVIEW READY";
      setTimeout(() => { sb.style.display = prevDisp; }, 1800);
    }
    
    
    const snapshotModal = document.getElementById("snapshot-modal");
    const btnSnapshotSave = document.getElementById("btn-snapshot-save");
    const btnSnapshotRetake = document.getElementById("btn-snapshot-retake");
    const btnSnapshotCloseX = document.getElementById("btn-snapshot-close-x");

    function closeSnapshotModal() {
      if (snapshotModal) snapshotModal.style.display = "none";
      if (state.lastSnapshotUrl) {
        URL.revokeObjectURL(state.lastSnapshotUrl);
        state.lastSnapshotUrl = null;
      }
    }

    if (btnSnapshotSave) {
      btnSnapshotSave.addEventListener("click", () => {
        if (state.lastSnapshotBlob) {
          const a = document.createElement("a");
          const dlUrl = URL.createObjectURL(state.lastSnapshotBlob);
          a.href = dlUrl;
          a.download = `divya-drusti-${Date.now()}.png`;
          a.click();
          setTimeout(() => URL.revokeObjectURL(dlUrl), 4000);
          const sb = document.getElementById("status-badge"), prevDisp = sb.style.display;
          sb.style.display = "block"; sb.textContent = "PHOTO SAVED TO DISK";
          setTimeout(() => { sb.style.display = prevDisp; }, 1800);
        }
        closeSnapshotModal();
      });
    }

    if (btnSnapshotRetake) btnSnapshotRetake.addEventListener("click", closeSnapshotModal);
    if (btnSnapshotCloseX) btnSnapshotCloseX.addEventListener("click", closeSnapshotModal);
    if (snapshotModal) {
      snapshotModal.addEventListener("click", e => {
        if (e.target === snapshotModal) closeSnapshotModal();
      });
    }


    document.getElementById("btn-capture").addEventListener("click", () => { state.capture = "manual"; settingsDrawer.classList.remove("open"); });
    document.getElementById("btn-hud-capture").addEventListener("click", () => { state.capture = "manual"; });

    // Global keyboard capture trigger (Space / C)
    window.addEventListener("keydown", e => {
      if (e.key === "Escape" && snapshotModal && snapshotModal.style.display !== "none") { closeSnapshotModal(); return; }
      if ((e.code === "Space" || e.key === "c" || e.key === "C") && e.target.tagName !== "INPUT" && e.target.tagName !== "TEXTAREA") {
        e.preventDefault();
        state.capture = "manual";
      }
    });
    document.getElementById("btn-fullscreen").addEventListener("click", () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen?.().catch(() => {});
    });
    let deferredInstall = null;
    const pwaBanner = document.getElementById("pwa-install-banner");
    const bannerInstallBtn = document.getElementById("btn-banner-install");
    const bannerDismissBtn = document.getElementById("btn-banner-dismiss");
    const rowInstall = document.getElementById("row-install");
    const iosModal = document.getElementById("ios-pwa-modal");
    const closeIosModalBtn = document.getElementById("btn-close-ios-modal");

    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

    // Detect standalone / installed PWA or APK wrapper mode
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches || 
                         window.navigator.standalone === true || 
                         (document.referrer && document.referrer.includes('android-app://'));

    // Suppress PWA install banner and row if running inside installed APK / PWA
    if (isStandalone) {
      if (pwaBanner) pwaBanner.style.display = "none";
      if (rowInstall) rowInstall.style.display = "none";
    } else {
      // If iOS, show install controls
      if (isIOS) {
        if (rowInstall) rowInstall.style.display = "flex";
        if (!localStorage.getItem("divya_pwa_dismissed") && pwaBanner) {
          pwaBanner.style.display = "flex";
        }
      }
    }

    window.addEventListener("beforeinstallprompt", e => {
      if (isStandalone) return;
      e.preventDefault();
      deferredInstall = e;
      if (rowInstall) rowInstall.style.display = "flex";
      if (!localStorage.getItem("divya_pwa_dismissed") && pwaBanner) {
        pwaBanner.style.display = "flex";
      }
    });

    const triggerInstallFlow = async () => {
      if (deferredInstall) {
        deferredInstall.prompt();
        const { outcome } = await deferredInstall.userChoice;
        deferredInstall = null;
        if (pwaBanner) pwaBanner.style.display = "none";
        if (rowInstall) rowInstall.style.display = "none";
        if (outcome === "accepted") {
          localStorage.setItem("divya_pwa_dismissed", "installed");
        }
      } else if (isIOS) {
        if (iosModal) iosModal.style.display = "flex";
      }
    };

    if (bannerInstallBtn) {
      bannerInstallBtn.addEventListener("click", triggerInstallFlow);
    }

    const btnInstall = document.getElementById("btn-install");
    if (btnInstall) {
      btnInstall.addEventListener("click", triggerInstallFlow);
    }

    if (bannerDismissBtn) {
      bannerDismissBtn.addEventListener("click", () => {
        if (pwaBanner) pwaBanner.style.display = "none";
        localStorage.setItem("divya_pwa_dismissed", "true");
      });
    }

    if (closeIosModalBtn) {
      closeIosModalBtn.addEventListener("click", () => {
        if (iosModal) iosModal.style.display = "none";
      });
    }

    window.addEventListener("appinstalled", () => {
      deferredInstall = null;
      if (pwaBanner) pwaBanner.style.display = "none";
      if (rowInstall) rowInstall.style.display = "none";
      localStorage.setItem("divya_pwa_dismissed", "installed");
    });
    
    /*
    /* =========================================================================
       AIR MUSIC ENGINE: POLYPHONIC FINGER PIANO & VIRTUAL HOLOGRAPHIC DRUMS
       ========================================================================= */
    const musicRipples = [];
    const activeFingerTouches = new Map(); // key: fingerId -> active target key/pad
    const activePointers = new Map(); // key: pointerId -> { x, y, isDown, isPointer }

    // Mouse & Multi-Touch screen interaction for Air Music
    function handleMusicPointerDown(e) {
      if (state.scene !== "MUSIC") return;
      if (e.target.closest("button, .interactive, .dock-container, .scene-pill, .modal, .settings-drawer, .tour-card, input, select")) return;

      if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();

      const rect = vecCanvas.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;

      const x = (e.clientX - rect.left) * (vecCanvas.width / rect.width);
      const y = (e.clientY - rect.top) * (vecCanvas.height / rect.height);
      activePointers.set(e.pointerId, { x, y, isDown: true, isPointer: true });
      if (e.cancelable) e.preventDefault();
    }

    function handleMusicPointerMove(e) {
      if (state.scene !== "MUSIC") return;
      if (!activePointers.has(e.pointerId)) return;

      const rect = vecCanvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (vecCanvas.width / rect.width);
      const y = (e.clientY - rect.top) * (vecCanvas.height / rect.height);
      activePointers.set(e.pointerId, { x, y, isDown: true, isPointer: true });
      if (e.cancelable) e.preventDefault();
    }

    function handleMusicPointerUp(e) {
      if (activePointers.has(e.pointerId)) {
        const pKey = `ptr_${e.pointerId}`;
        for (const [k, v] of activeFingerTouches.entries()) {
          if (k.includes(pKey)) activeFingerTouches.set(k, false);
        }
        activePointers.delete(e.pointerId);
      }
    }

    window.addEventListener("pointerdown", handleMusicPointerDown, { passive: false });
    window.addEventListener("pointermove", handleMusicPointerMove, { passive: false });
    window.addEventListener("pointerup", handleMusicPointerUp, { passive: true });
    window.addEventListener("pointercancel", handleMusicPointerUp, { passive: true });

    // 1. Polyphonic Piano Synthesizer
    function playPianoNote(freq, velocity = 0.8) {
      if (state.soundEnabled === false) return;
      try {
        if (audioCtx.state === "suspended") audioCtx.resume();
        const now = audioCtx.currentTime;

        // Fundamental oscillator (warm sine)
        const osc1 = audioCtx.createOscillator();
        const gain1 = audioCtx.createGain();
        osc1.type = "sine";
        osc1.frequency.setValueAtTime(freq, now);
        gain1.gain.setValueAtTime(0.35 * velocity, now);
        gain1.gain.exponentialRampToValueAtTime(0.0001, now + 0.85);
        osc1.connect(gain1);
        gain1.connect(audioCtx.destination);
        osc1.start(now);
        osc1.stop(now + 0.86);

        // First harmonic (triangle warmth)
        const osc2 = audioCtx.createOscillator();
        const gain2 = audioCtx.createGain();
        osc2.type = "triangle";
        osc2.frequency.setValueAtTime(freq * 2, now);
        gain2.gain.setValueAtTime(0.12 * velocity, now);
        gain2.gain.exponentialRampToValueAtTime(0.0001, now + 0.55);
        osc2.connect(gain2);
        gain2.connect(audioCtx.destination);
        osc2.start(now);
        osc2.stop(now + 0.56);

        // Strike brilliance (transient attack)
        const osc3 = audioCtx.createOscillator();
        const gain3 = audioCtx.createGain();
        osc3.type = "sine";
        osc3.frequency.setValueAtTime(freq * 3, now);
        gain3.gain.setValueAtTime(0.08 * velocity, now);
        gain3.gain.exponentialRampToValueAtTime(0.0001, now + 0.15);
        osc3.connect(gain3);
        gain3.connect(audioCtx.destination);
        osc3.start(now);
        osc3.stop(now + 0.16);
      } catch (e) {
        console.warn("Piano audio error:", e);
      }
    }

    // 2. Holographic Drum Kit Synthesizer (Kick, Snare, Hi-Hat, Crash, Toms)
    function playDrumSound(type) {
      if (state.soundEnabled === false) return;
      try {
        if (audioCtx.state === "suspended") audioCtx.resume();
        const now = audioCtx.currentTime;

        if (type === "KICK") {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.type = "sine";
          osc.frequency.setValueAtTime(140, now);
          osc.frequency.exponentialRampToValueAtTime(38, now + 0.12);
          gain.gain.setValueAtTime(0.65, now);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.38);
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start(now);
          osc.stop(now + 0.4);
        } else if (type === "SNARE") {
          // Tone core
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.type = "triangle";
          osc.frequency.setValueAtTime(185, now);
          osc.frequency.exponentialRampToValueAtTime(80, now + 0.08);
          gain.gain.setValueAtTime(0.4, now);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.14);
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start(now);
          osc.stop(now + 0.15);

          // Snare wire rattle
          const bSize = Math.floor(audioCtx.sampleRate * 0.18);
          const buf = audioCtx.createBuffer(1, bSize, audioCtx.sampleRate);
          const data = buf.getChannelData(0);
          for (let i = 0; i < bSize; i++) data[i] = Math.random() * 2 - 1;
          const noise = audioCtx.createBufferSource();
          noise.buffer = buf;
          const flt = audioCtx.createBiquadFilter();
          flt.type = "highpass";
          flt.frequency.value = 1200;
          const ng = audioCtx.createGain();
          ng.gain.setValueAtTime(0.35, now);
          ng.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);
          noise.connect(flt);
          flt.connect(ng);
          ng.connect(audioCtx.destination);
          noise.start(now);
        } else if (type === "HIHAT") {
          const bSize = Math.floor(audioCtx.sampleRate * 0.07);
          const buf = audioCtx.createBuffer(1, bSize, audioCtx.sampleRate);
          const data = buf.getChannelData(0);
          for (let i = 0; i < bSize; i++) data[i] = Math.random() * 2 - 1;
          const noise = audioCtx.createBufferSource();
          noise.buffer = buf;
          const flt = audioCtx.createBiquadFilter();
          flt.type = "highpass";
          flt.frequency.value = 7500;
          const gain = audioCtx.createGain();
          gain.gain.setValueAtTime(0.28, now);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.07);
          noise.connect(flt);
          flt.connect(gain);
          gain.connect(audioCtx.destination);
          noise.start(now);
        } else if (type === "CRASH") {
          const bSize = Math.floor(audioCtx.sampleRate * 0.9);
          const buf = audioCtx.createBuffer(1, bSize, audioCtx.sampleRate);
          const data = buf.getChannelData(0);
          for (let i = 0; i < bSize; i++) data[i] = Math.random() * 2 - 1;
          const noise = audioCtx.createBufferSource();
          noise.buffer = buf;
          const flt = audioCtx.createBiquadFilter();
          flt.type = "bandpass";
          flt.frequency.value = 5200;
          flt.Q.value = 1.4;
          const gain = audioCtx.createGain();
          gain.gain.setValueAtTime(0.38, now);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.9);
          noise.connect(flt);
          flt.connect(gain);
          gain.connect(audioCtx.destination);
          noise.start(now);
        } else if (type === "TOM_HIGH") {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.frequency.setValueAtTime(240, now);
          osc.frequency.exponentialRampToValueAtTime(130, now + 0.22);
          gain.gain.setValueAtTime(0.45, now);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.25);
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start(now);
          osc.stop(now + 0.26);
        } else if (type === "TOM_LOW") {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.frequency.setValueAtTime(135, now);
          osc.frequency.exponentialRampToValueAtTime(75, now + 0.3);
          gain.gain.setValueAtTime(0.5, now);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.32);
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start(now);
          osc.stop(now + 0.33);
        }
      } catch (e) {
        console.warn("Drum audio error:", e);
      }
    }

// 3. Acoustic Air Guitar Synthesizer (6 strings, physical harmonic ringout)
    const GUITAR_BASE_FREQS = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]; // E2, A2, D3, G3, B3, E4
    const GUITAR_STRING_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"];
    const GUITAR_CHORDS = {
      "OPEN": { name: "OPEN", frets: [0, 0, 0, 0, 0, 0] },
      "EM":   { name: "Em",   frets: [0, 2, 2, 0, 0, 0] },
      "G":    { name: "G",    frets: [3, 2, 0, 0, 3, 3] },
      "C":    { name: "C",    frets: [-1, 3, 2, 0, 1, 0] },
      "D":    { name: "D",    frets: [-1, -1, 0, 2, 3, 2] },
      "AM":   { name: "Am",   frets: [-1, 0, 2, 2, 1, 0] },
      "F":    { name: "F",    frets: [1, 3, 3, 2, 1, 1] }
    };

    function playGuitarPluck(stringIdx, fret = 0, velocity = 1.0) {
      if (state.soundEnabled === false || fret < 0) return;
      try {
        if (audioCtx.state === "suspended") audioCtx.resume();
        const now = audioCtx.currentTime;
        const baseFreq = GUITAR_BASE_FREQS[stringIdx] || 110.0;
        const freq = baseFreq * Math.pow(2, fret / 12);
        const dur = 1.2 + (5 - stringIdx) * 0.25;

        const osc1 = audioCtx.createOscillator();
        const osc2 = audioCtx.createOscillator();
        const filter = audioCtx.createBiquadFilter();
        const gain = audioCtx.createGain();

        osc1.type = "triangle";
        osc1.frequency.setValueAtTime(freq, now);

        osc2.type = "sawtooth";
        osc2.frequency.setValueAtTime(freq * 1.002, now);

        filter.type = "lowpass";
        filter.frequency.setValueAtTime(Math.min(8000, freq * 8), now);
        filter.frequency.exponentialRampToValueAtTime(Math.min(3000, freq * 1.8), now + 0.35);
        filter.Q.value = 3.5;

        const vol = Math.min(0.55, 0.38 * velocity);
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.exponentialRampToValueAtTime(vol, now + 0.004);
        gain.gain.exponentialRampToValueAtTime(vol * 0.6, now + 0.12);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + dur);

        osc1.connect(filter);
        osc2.connect(filter);
        filter.connect(gain);
        gain.connect(audioCtx.destination);

        osc1.start(now);
        osc2.start(now);
        osc1.stop(now + dur);
        osc2.stop(now + dur);
      } catch (e) {
        console.warn("Guitar audio error:", e);
      }
    }

    // 3. Main Air Music Draw Loop
    function drawMusic(w, h) {
      const nowMs = performance.now();
      badge("AIR MUSIC · play with fingers, mouse or touch · polyphonic synthesizers");
      const c = vCtx;
      const instrument = state.instrument || "PIANO";
      const showPiano = instrument === "PIANO" || instrument === "DUAL";
      const showDrums = instrument === "DRUMS" || instrument === "DUAL";

      // Collect all fingertips from detected hands
      const fingertips = [];
      if (state.handData && state.handData.length) {
        state.handData.forEach((hd, handIdx) => {
          [4, 8, 12, 16, 20].forEach(tipIdx => {
            const pt = landmarkToScreen(hd.landmarks[tipIdx]);
            fingertips.push({
              id: `${handIdx}_${tipIdx}`,
              x: pt.x,
              y: pt.y,
              handIdx,
              tipIdx,
              isPointer: false
            });
          });
        });
      }

      // Collect all active mouse clicks & touchscreen touches
      for (const [ptrId, ptr] of activePointers.entries()) {
        if (ptr.isDown) {
          fingertips.push({
            id: `ptr_${ptrId}`,
            x: ptr.x,
            y: ptr.y,
            isPointer: true
          });
        }
      }

      // ----------------------------------------------------
      // A. DRUMS RENDERING & STRIKE COLLISION
      // ----------------------------------------------------
      if (showDrums) {
        const drumYOffset = instrument === "DUAL" ? -0.06 : 0;
        const drumScale = instrument === "DUAL" ? 0.85 : 1.0;

        const drumPads = [
          { type: "CRASH", label: "CRASH", x: w * 0.18, y: h * (0.28 + drumYOffset), r: 46 * drumScale, color: "#fbbf24", glow: "rgba(251, 191, 36, 0.4)" },
          { type: "HIHAT", label: "HI-HAT", x: w * 0.36, y: h * (0.22 + drumYOffset), r: 40 * drumScale, color: "#38bdf8", glow: "rgba(56, 189, 248, 0.4)" },
          { type: "TOM_HIGH", label: "HI TOM", x: w * 0.54, y: h * (0.22 + drumYOffset), r: 42 * drumScale, color: "#a855f7", glow: "rgba(168, 85, 247, 0.4)" },
          { type: "CRASH", label: "RIDE", x: w * 0.82, y: h * (0.28 + drumYOffset), r: 46 * drumScale, color: "#f59e0b", glow: "rgba(245, 158, 11, 0.4)" },
          { type: "SNARE", label: "SNARE", x: w * 0.30, y: h * (0.45 + drumYOffset), r: 52 * drumScale, color: "#f43f5e", glow: "rgba(244, 63, 94, 0.45)" },
          { type: "TOM_LOW", label: "FLOOR TOM", x: w * 0.70, y: h * (0.45 + drumYOffset), r: 52 * drumScale, color: "#10b981", glow: "rgba(16, 185, 129, 0.4)" },
          { type: "KICK", label: "BASS KICK", x: w * 0.50, y: h * (0.50 + drumYOffset), r: 60 * drumScale, color: "#6366f1", glow: "rgba(99, 102, 241, 0.5)" }
        ];

        drumPads.forEach(pad => {
          let hitByThisFrame = false;

          fingertips.forEach(ft => {
            const dist = Math.hypot(ft.x - pad.x, ft.y - pad.y);
            const isInside = dist < pad.r;
            const touchKey = `drum_${ft.id}_${pad.type}`;

            if (isInside) {
              if (!activeFingerTouches.get(touchKey)) {
                // Strike triggered!
                activeFingerTouches.set(touchKey, true);
                playDrumSound(pad.type);
                hitByThisFrame = true;

                // Visual ripples
                musicRipples.push({
                  x: pad.x,
                  y: pad.y,
                  color: pad.color,
                  radius: pad.r * 0.8,
                  maxRadius: pad.r * 1.8,
                  alpha: 0.95
                });
              } else {
                hitByThisFrame = true;
              }
            } else {
              activeFingerTouches.set(touchKey, false);
            }
          });

          // Draw Drum Pad
          c.save();
          // Pad frosted base
          c.beginPath();
          c.arc(pad.x, pad.y, pad.r, 0, TAU);
          c.fillStyle = hitByThisFrame ? pad.color : "rgba(15, 23, 42, 0.72)";
          c.fill();

          // Outer glowing ring
          c.lineWidth = hitByThisFrame ? 4 : 2;
          c.strokeStyle = pad.color;
          c.shadowColor = pad.color;
          c.shadowBlur = hitByThisFrame ? 22 : 8;
          c.stroke();

          // Inner resonance ring
          c.beginPath();
          c.arc(pad.x, pad.y, pad.r * 0.65, 0, TAU);
          c.lineWidth = 1.2;
          c.strokeStyle = "rgba(255, 255, 255, 0.35)";
          c.stroke();

          // Pad Label
          c.font = `700 ${Math.max(9, Math.round(pad.r * 0.22))}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
          c.fillStyle = hitByThisFrame ? "#000000" : "#f1f5f9";
          c.textAlign = "center";
          c.textBaseline = "middle";
          c.fillText(pad.label, pad.x, pad.y);
          c.restore();
        });
      }

      // ----------------------------------------------------
      // B. PIANO KEYBOARD RENDERING & KEY PRESS COLLISION
      // ----------------------------------------------------
      if (showPiano) {
        const baseOct = 4 + (state.octaveOffset || 0);
        const nextOct = baseOct + 1;
        const whiteNotes = [`C${baseOct}`, `D${baseOct}`, `E${baseOct}`, `F${baseOct}`, `G${baseOct}`, `A${baseOct}`, `B${baseOct}`, `C${nextOct}`, `D${nextOct}`, `E${nextOct}`];
        const whiteFreqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25];

        const blackNoteMap = [
          { idx: 0, note: `C#${baseOct}`, freq: 277.18 },
          { idx: 1, note: `D#${baseOct}`, freq: 311.13 },
          { idx: 3, note: `F#${baseOct}`, freq: 369.99 },
          { idx: 4, note: `G#${baseOct}`, freq: 415.30 },
          { idx: 5, note: `A#${baseOct}`, freq: 466.16 },
          { idx: 7, note: `C#${nextOct}`, freq: 554.37 },
          { idx: 8, note: `D#${nextOct}`, freq: 622.25 }
        ];

        const octMult = Math.pow(2, state.octaveOffset || 0);

        let kbY;
        if (instrument === "DUAL") {
          kbY = h * 0.68;
        } else {
          const lvl = state.pianoLevel || "FLOATING";
          kbY = lvl === "LOW" ? h * 0.62 : (lvl === "HIGH" ? h * 0.30 : h * 0.44);
        }
        const kbH = instrument === "DUAL" ? h * 0.26 : h * 0.35;
        const numWhite = whiteNotes.length;
        const whiteW = w / numWhite;
        const blackW = whiteW * 0.58;
        const blackH = kbH * 0.62;

        // Construct Black Keys Geometry
        const blackKeys = blackNoteMap.map(bk => ({
          note: bk.note,
          freq: bk.freq * octMult,
          x: (bk.idx + 1) * whiteW - blackW / 2,
          y: kbY,
          w: blackW,
          h: blackH
        }));

        // Construct White Keys Geometry
        const whiteKeys = whiteNotes.map((n, idx) => ({
          note: n,
          freq: whiteFreqs[idx] * octMult,
          x: idx * whiteW,
          y: kbY,
          w: whiteW,
          h: kbH
        }));

        // Detect collisions: Check Black keys first (top layer), then White keys
        const hitNotesThisFrame = new Set();

        fingertips.forEach(ft => {
          let hitKey = null;

          // Check black keys
          for (const bk of blackKeys) {
            if (ft.x >= bk.x && ft.x <= bk.x + bk.w && ft.y >= bk.y && ft.y <= bk.y + bk.h) {
              hitKey = bk;
              break;
            }
          }

          // If not black key, check white keys
          if (!hitKey) {
            for (const wk of whiteKeys) {
              if (ft.x >= wk.x && ft.x <= wk.x + wk.w && ft.y >= wk.y && ft.y <= wk.y + wk.h) {
                hitKey = wk;
                break;
              }
            }
          }

          if (hitKey) {
            hitNotesThisFrame.add(hitKey.note);
            const touchKey = `piano_${ft.id}_${hitKey.note}`;
            if (!activeFingerTouches.get(touchKey)) {
              activeFingerTouches.set(touchKey, true);
              playPianoNote(hitKey.freq, 0.9);

              musicRipples.push({
                x: ft.x,
                y: ft.y,
                color: "#38bdf8",
                radius: 12,
                maxRadius: 55,
                alpha: 0.95
              });
            }
          } else {
            // Finger not touching any key
            for (const [k, v] of activeFingerTouches.entries()) {
              if (k.startsWith(`piano_${ft.id}_`)) activeFingerTouches.set(k, false);
            }
          }
        });

        // 1. Draw White Keys
        whiteKeys.forEach(wk => {
          const isPressed = hitNotesThisFrame.has(wk.note);
          c.save();
          c.fillStyle = isPressed ? "rgba(56, 189, 248, 0.85)" : "rgba(240, 244, 250, 0.88)";
          c.strokeStyle = "rgba(15, 23, 42, 0.5)";
          c.lineWidth = 1.5;

          // Rounded bottom corners
          const pad = 1.5;
          c.beginPath();
          c.rect(wk.x + pad, wk.y, wk.w - pad * 2, wk.h);
          c.fill();
          c.stroke();

          // Active glow
          if (isPressed) {
            c.shadowColor = "#38bdf8";
            c.shadowBlur = 18;
            c.strokeStyle = "#ffffff";
            c.stroke();
          }

          // Note Label at bottom
          c.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
          c.fillStyle = isPressed ? "#ffffff" : "#334155";
          c.textAlign = "center";
          c.textBaseline = "bottom";
          c.fillText(wk.note, wk.x + wk.w / 2, wk.y + wk.h - 8);
          c.restore();
        });

        // 2. Draw Black Keys
        blackKeys.forEach(bk => {
          const isPressed = hitNotesThisFrame.has(bk.note);
          c.save();
          c.fillStyle = isPressed ? "#38bdf8" : "rgba(15, 23, 42, 0.95)";
          c.strokeStyle = isPressed ? "#ffffff" : "rgba(148, 163, 184, 0.35)";
          c.lineWidth = 1.2;

          c.beginPath();
          c.rect(bk.x, bk.y, bk.w, bk.h);
          c.fill();
          c.stroke();

          if (isPressed) {
            c.shadowColor = "#38bdf8";
            c.shadowBlur = 20;
            c.stroke();
          }

          // Black key label
          c.font = '600 9px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
          c.fillStyle = isPressed ? "#000000" : "#94a3b8";
          c.textAlign = "center";
          c.textBaseline = "bottom";
          c.fillText(bk.note.slice(0, 2), bk.x + bk.w / 2, bk.y + bk.h - 6);
          c.restore();
        });
      }

      // ----------------------------------------------------
      // C. FINGERTIP CURSORS & SPATIAL RIPPLES
      // ----------------------------------------------------
      // Draw Expanding Musical Ripples
      for (let i = musicRipples.length - 1; i >= 0; i--) {
        const rp = musicRipples[i];
        rp.radius += 2.4;
        rp.alpha -= 0.045;
        if (rp.alpha <= 0) {
          musicRipples.splice(i, 1);
          continue;
        }

        c.save();
        c.strokeStyle = rp.color;
        c.lineWidth = 2.2;
        c.globalAlpha = rp.alpha;
        c.shadowColor = rp.color;
        c.shadowBlur = 10;
        c.beginPath();
        c.arc(rp.x, rp.y, rp.radius, 0, TAU);
        c.stroke();
        c.restore();
      }

      // ----------------------------------------------------
      // C. AIR GUITAR RENDERING & STRUM COLLISION
      // ----------------------------------------------------
      if (instrument === "GUITAR") {
        const chordKeys = Object.keys(GUITAR_CHORDS);
        const activeChordKey = state.activeGuitarChord || "OPEN";
        const curChord = GUITAR_CHORDS[activeChordKey] || GUITAR_CHORDS["OPEN"];
        const isMobile = w < 640;

        let strLeft, strRight, strTopY, strSpacing, fretboardH;

        if (isMobile) {
          // Mobile / Portrait Layout: Responsive Horizontal Chord Strip
          const chordBarY = Math.floor(h * 0.28);
          const chordBarH = 34;
          const chordBtnW = Math.max(48, Math.floor((w - 28) / chordKeys.length));

          c.save();
          c.fillStyle = "rgba(10, 15, 26, 0.88)";
          c.strokeStyle = "rgba(56, 189, 248, 0.35)";
          c.lineWidth = 1;
          c.beginPath();
          c.roundRect(10, chordBarY - 6, w - 20, chordBarH + 12, 10);
          c.fill();
          c.stroke();

          chordKeys.forEach((ck, ci) => {
            const bx = 14 + ci * chordBtnW;
            const bw = chordBtnW - 4;
            const isAct = ck === activeChordKey;
            c.fillStyle = isAct ? "rgba(56, 189, 248, 0.32)" : "rgba(255, 255, 255, 0.05)";
            c.strokeStyle = isAct ? "#38bdf8" : "rgba(255, 255, 255, 0.12)";
            c.lineWidth = isAct ? 1.5 : 1;
            c.beginPath();
            c.roundRect(bx, chordBarY, bw, chordBarH, 6);
            c.fill();
            c.stroke();

            c.font = "700 11px system-ui, -apple-system, sans-serif";
            c.fillStyle = isAct ? "#ffffff" : "#94a3b8";
            c.textAlign = "center";
            c.textBaseline = "middle";
            const shortName = ck === "OPEN" ? "OPEN" : (ck.replace("_MAJ", "").replace("_MIN", "m"));
            c.fillText(shortName, bx + bw / 2, chordBarY + chordBarH / 2);

            fingertips.forEach(ft => {
              if (ft.x >= bx && ft.x <= bx + bw && ft.y >= chordBarY && ft.y <= chordBarY + chordBarH) {
                if (state.activeGuitarChord !== ck) {
                  state.activeGuitarChord = ck;
                  playAudioTone(440, "sine", 0.04, 0.1);
                  badge(`AIR GUITAR · CHORD ${GUITAR_CHORDS[ck].name}`);
                }
              }
            });
          });
          c.restore();

          strLeft = 36;
          strRight = w - 20;
          strTopY = chordBarY + chordBarH + 20;
          strSpacing = Math.min(48, Math.floor((h * 0.40) / 5));
          fretboardH = strSpacing * 5 + 32;
        } else {
          // Desktop / Landscape Layout: Sleek Chord Panel on Left
          const chordPanelW = Math.min(130, Math.floor(w * 0.16));
          const chordBtnH = 34;
          const chordStartY = Math.floor(h * 0.32);

          c.save();
          c.fillStyle = "rgba(10, 15, 26, 0.88)";
          c.strokeStyle = "rgba(56, 189, 248, 0.35)";
          c.lineWidth = 1;
          c.beginPath();
          c.roundRect(12, chordStartY - 10, chordPanelW, chordKeys.length * (chordBtnH + 6) + 16, 10);
          c.fill();
          c.stroke();

          chordKeys.forEach((ck, ci) => {
            const cy = chordStartY + ci * (chordBtnH + 6);
            const isAct = ck === activeChordKey;
            c.fillStyle = isAct ? "rgba(56, 189, 248, 0.28)" : "rgba(255, 255, 255, 0.04)";
            c.strokeStyle = isAct ? "#38bdf8" : "rgba(255, 255, 255, 0.12)";
            c.lineWidth = isAct ? 1.5 : 1;
            c.beginPath();
            c.roundRect(18, cy, chordPanelW - 12, chordBtnH, 6);
            c.fill();
            c.stroke();

            c.font = "700 12px system-ui, -apple-system, sans-serif";
            c.fillStyle = isAct ? "#ffffff" : "#94a3b8";
            c.textAlign = "center";
            c.textBaseline = "middle";
            c.fillText(GUITAR_CHORDS[ck].name, 18 + (chordPanelW - 12) / 2, cy + chordBtnH / 2);

            fingertips.forEach(ft => {
              if (ft.x >= 18 && ft.x <= 18 + chordPanelW - 12 && ft.y >= cy && ft.y <= cy + chordBtnH) {
                if (state.activeGuitarChord !== ck) {
                  state.activeGuitarChord = ck;
                  playAudioTone(440, "sine", 0.04, 0.1);
                  badge(`AIR GUITAR · CHORD ${GUITAR_CHORDS[ck].name}`);
                }
              }
            });
          });
          c.restore();

          strLeft = chordPanelW + 36;
          strRight = w - 24;
          strTopY = chordStartY;
          strSpacing = Math.min(50, Math.floor((h * 0.40) / 5));
          fretboardH = strSpacing * 5 + 32;
        }

        const strW = strRight - strLeft;

        if (!state.guitarStringVibrations) {
          state.guitarStringVibrations = [0, 0, 0, 0, 0, 0].map(() => ({ amp: 0, lastPluck: 0 }));
        }

        // Draw Fretboard Body (dark acoustic neck background under strings)
        c.save();
        const fretboardBg = c.createLinearGradient(strLeft, strTopY, strRight, strTopY);
        fretboardBg.addColorStop(0, "rgba(15, 23, 42, 0.85)");
        fretboardBg.addColorStop(0.5, "rgba(20, 29, 50, 0.88)");
        fretboardBg.addColorStop(1, "rgba(15, 23, 42, 0.85)");
        c.fillStyle = fretboardBg;
        c.strokeStyle = "rgba(56, 189, 248, 0.28)";
        c.lineWidth = 1.5;
        c.beginPath();
        c.roundRect(strLeft - 16, strTopY - 14, strW + 32, fretboardH, 12);
        c.fill();
        c.stroke();

        // Vertical Fret Wires and Inlay Dots
        const numFrets = 6;
        for (let f = 1; f <= numFrets; f++) {
          const fx = strLeft + (f / (numFrets + 1)) * strW;
          c.strokeStyle = "rgba(148, 163, 184, 0.25)";
          c.lineWidth = 1.2;
          c.beginPath();
          c.moveTo(fx, strTopY - 10);
          c.lineTo(fx, strTopY - 10 + fretboardH - 8);
          c.stroke();

          // Mother-of-pearl fret markers at 3rd and 5th fret
          if (f === 2 || f === 4) {
            c.fillStyle = "rgba(255, 255, 255, 0.35)";
            c.beginPath();
            c.arc(fx - (strW / (numFrets + 1)) * 0.5, strTopY + fretboardH * 0.44, 3.5, 0, TAU);
            c.fill();
          }
        }

        // Nut Bar (Left anchor)
        c.fillStyle = "rgba(226, 232, 240, 0.6)";
        c.beginPath();
        c.roundRect(strLeft - 6, strTopY - 10, 4, fretboardH - 8, 2);
        c.fill();

        // Bridge Bar (Right anchor)
        c.fillStyle = "rgba(245, 158, 11, 0.5)";
        c.beginPath();
        c.roundRect(strRight + 2, strTopY - 10, 4, fretboardH - 8, 2);
        c.fill();
        c.restore();

        // 6 Horizontal Glowing Guitar Strings
        const stringColors = ["#f59e0b", "#fbbf24", "#38bdf8", "#818cf8", "#a855f7", "#ec4899"];
        const stringThicknesses = [3.8, 3.2, 2.7, 2.2, 1.8, 1.4];

        for (let sIdx = 0; sIdx < 6; sIdx++) {
          const stringY = strTopY + sIdx * strSpacing;
          const vib = state.guitarStringVibrations[sIdx];
          const fretOffset = curChord.frets[sIdx];
          const isMuted = fretOffset < 0;

          // Check collisions with fingertips & pointers (smooth touch/mouse strumming)
          fingertips.forEach(ft => {
            if (ft.x >= strLeft - 14 && ft.x <= strRight + 14) {
              const dy = Math.abs(ft.y - stringY);
              const touchKey = `gtr_${ft.id}_${sIdx}`;
              if (dy < 20) {
                if (!activeFingerTouches.get(touchKey) && (nowMs - vib.lastPluck > 70)) {
                  activeFingerTouches.set(touchKey, true);
                  vib.lastPluck = nowMs;
                  vib.amp = 16.0;

                  if (!isMuted) {
                    playGuitarPluck(sIdx, fretOffset);
                    badge(`AIR GUITAR · ${GUITAR_CHORDS[activeChordKey].name} · STRING ${GUITAR_STRING_NAMES[sIdx]}`);
                    musicRipples.push({
                      x: ft.x,
                      y: stringY,
                      color: stringColors[sIdx],
                      radius: 10,
                      maxRadius: 42,
                      alpha: 0.95
                    });
                  } else {
                    playAudioTone(180, "sawtooth", 0.04, 0.08);
                  }
                }
              } else {
                activeFingerTouches.set(touchKey, false);
              }
            }
          });

          const dt = (nowMs - vib.lastPluck) / 1000;
          const curAmp = Math.max(0, vib.amp * Math.exp(-dt * 4.5));
          const strThickness = stringThicknesses[sIdx];

          c.save();
          c.lineWidth = strThickness;
          c.strokeStyle = isMuted ? "rgba(100, 116, 139, 0.45)" : stringColors[sIdx];
          c.shadowColor = isMuted ? "transparent" : stringColors[sIdx];
          c.shadowBlur = curAmp > 1 ? 14 : 3;

          c.beginPath();
          if (curAmp > 0.4) {
            const steps = 36;
            for (let i = 0; i <= steps; i++) {
              const u = i / steps;
              const px = strLeft + u * strW;
              const envelope = Math.sin(u * Math.PI);
              const wave = Math.sin(u * Math.PI * 2 + nowMs * 0.05) * curAmp * envelope;
              const py = stringY + wave;
              i === 0 ? c.moveTo(px, py) : c.lineTo(px, py);
            }
          } else {
            c.moveTo(strLeft, stringY);
            c.lineTo(strRight, stringY);
          }
          c.stroke();

          // String Note Badge / Tag on Left Nut
          c.font = "700 11px system-ui, -apple-system, sans-serif";
          c.fillStyle = isMuted ? "#64748b" : (curAmp > 0.5 ? "#ffffff" : "#cbd5e1");
          c.textAlign = "right";
          c.textBaseline = "middle";
          const noteLabel = isMuted ? "X" : (fretOffset > 0 ? `${GUITAR_STRING_NAMES[sIdx]}+${fretOffset}` : GUITAR_STRING_NAMES[sIdx]);
          c.fillText(noteLabel, strLeft - 10, stringY);
          c.restore();
        }
      }

      // Draw Glowing Halos on all active fingertips
      fingertips.forEach(ft => {
        c.save();
        glow(c, ft.x, ft.y, 14, "rgba(56, 189, 248, 0.8)", "rgba(56, 189, 248, 0)");
        c.fillStyle = "#ffffff";
        c.beginPath();
        c.arc(ft.x, ft.y, 4, 0, TAU);
        c.fill();
        c.restore();
      });
    }

    // 4. Air Music Dock Tab Listeners
    document.querySelectorAll("[data-instrument]").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-instrument]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.instrument = btn.dataset.instrument;
        const sb = document.getElementById("status-badge");
        if (sb) sb.textContent = `AIR MUSIC: ${state.instrument}`;
      });
    });

    const btnMusicLevel = document.getElementById("btn-music-level");
    if (btnMusicLevel) {
      btnMusicLevel.addEventListener("click", () => {
        if (!state.pianoLevel || state.pianoLevel === "FLOATING") {
          state.pianoLevel = "LOW";
          btnMusicLevel.textContent = "Position: Desk";
        } else if (state.pianoLevel === "LOW") {
          state.pianoLevel = "HIGH";
          btnMusicLevel.textContent = "Position: High";
        } else {
          state.pianoLevel = "FLOATING";
          btnMusicLevel.textContent = "Position: Floating";
        }
        const sb = document.getElementById("status-badge");
        if (sb) sb.textContent = `AIR MUSIC: ${state.instrument || "PIANO"} (${state.pianoLevel})`;
      });
    }

    const btnMusicOctave = document.getElementById("btn-music-octave");
    if (btnMusicOctave) {
      btnMusicOctave.addEventListener("click", () => {
        state.octaveOffset = state.octaveOffset === 0 ? 1 : (state.octaveOffset === 1 ? -1 : 0);
        const curBase = 4 + state.octaveOffset;
        btnMusicOctave.textContent = `Octave: C${curBase}–E${curBase + 1}`;
        badge(`AIR PIANO · OCTAVE C${curBase}–E${curBase + 1}`);
      });
    }

    const btnMusicClear = document.getElementById("btn-music-clear");
    if (btnMusicClear) {
      btnMusicClear.addEventListener("click", () => {
        musicRipples.length = 0;
        activeFingerTouches.clear();
      });
    }


    /* =========================================================================
       SPOTLIGHT TOUR ENGINE (First-time walkthrough & Settings launchable)
       ========================================================================= */
    const TOUR_STEPS = [
      {
        target: ".scene-selector",
        title: "Modes & Studio Workspaces",
        desc: "Tap any tab to switch between Spatial Prism, AI Hair & Makeup, Mystic Jutsu, Prana Catch Game, or 3D Avatar.",
        position: "top"
      },
      {
        target: ".dock-container:not([style*='display: none'])",
        fallbackTarget: "#prism-dock",
        title: "Real-Time Effects & Shaders",
        desc: "Swipe horizontally along the effects dock to choose from 17 real-time GPU shaders and Jutsu spells.",
        position: "top"
      },
      {
        target: "#viewport-frame",
        title: "Touchless Spatial Gestures",
        desc: "Dual-pinch thumbs and index fingers to shape the prism. Tap Capture in the header or press Space / C to take snapshots.",
        position: "center"
      },
      {
        target: "#btn-open-settings",
        title: "Settings & Camera Controls",
        desc: "Flip camera, toggle mirror, and open Settings for precision sliders, flock density, and tracking overlays.",
        position: "bottom"
      }
    ];

    let currentTourStep = 0;
    const tourOverlay = document.getElementById("tour-overlay");
    const tourHighlight = document.getElementById("tour-highlight");
    const tourCard = document.getElementById("tour-card");
    const tourStepBadge = document.getElementById("tour-step-badge");
    const tourTitle = document.getElementById("tour-title");
    const tourDesc = document.getElementById("tour-desc");
    const btnTourPrev = document.getElementById("btn-tour-prev");
    const btnTourNext = document.getElementById("btn-tour-next");
    const btnTourClose = document.getElementById("btn-tour-close");
    const btnRestartTour = document.getElementById("btn-restart-tour");

    function updateTourCard() {
      if (!tourOverlay || tourOverlay.style.display === "none") return;
      const step = TOUR_STEPS[currentTourStep];
      if (!step) return;

      let el = document.querySelector(step.target);
      if (!el && step.fallbackTarget) el = document.querySelector(step.fallbackTarget);
      if (!el) el = document.getElementById("stage-container");

      if (tourStepBadge) tourStepBadge.textContent = `Step ${currentTourStep + 1} of ${TOUR_STEPS.length}`;
      if (tourTitle) tourTitle.textContent = step.title;
      if (tourDesc) tourDesc.textContent = step.desc;

      document.querySelectorAll(".tour-dot").forEach((dot, idx) => {
        dot.classList.toggle("active", idx === currentTourStep);
      });

      if (btnTourPrev) btnTourPrev.style.display = currentTourStep === 0 ? "none" : "block";
      if (btnTourNext) btnTourNext.textContent = currentTourStep === TOUR_STEPS.length - 1 ? "Finish" : "Next";

      const rect = el.getBoundingClientRect();
      const pad = 6;
      const hlLeft = Math.max(0, rect.left - pad);
      const hlTop = Math.max(0, rect.top - pad);
      const hlWidth = rect.width + pad * 2;
      const hlHeight = rect.height + pad * 2;

      if (tourHighlight) {
        tourHighlight.style.left = `${hlLeft}px`;
        tourHighlight.style.top = `${hlTop}px`;
        tourHighlight.style.width = `${hlWidth}px`;
        tourHighlight.style.height = `${hlHeight}px`;
      }

      if (tourCard) {
        const cardW = tourCard.offsetWidth || 320;
        const cardH = tourCard.offsetHeight || 140;

        if (step.position === "center") {
          tourCard.style.left = "50%";
          tourCard.style.top = "50%";
          tourCard.style.transform = "translate(-50%, -50%)";
        } else if (step.position === "top") {
          const top = Math.max(16, rect.top - cardH - 14);
          const left = Math.max(14, Math.min(window.innerWidth - cardW - 14, rect.left + rect.width / 2 - cardW / 2));
          tourCard.style.left = `${left}px`;
          tourCard.style.top = `${top}px`;
          tourCard.style.transform = "none";
        } else {
          const top = Math.min(window.innerHeight - cardH - 16, rect.bottom + 14);
          const left = Math.max(14, Math.min(window.innerWidth - cardW - 14, rect.right - cardW));
          tourCard.style.left = `${left}px`;
          tourCard.style.top = `${top}px`;
          tourCard.style.transform = "none";
        }
      }
    }

    function startTour() {
      currentTourStep = 0;
      try { localStorage.setItem("divya_tour_completed", "true"); } catch (e) {}
      if (tourOverlay) {
        tourOverlay.style.display = "block";
        setTimeout(updateTourCard, 60);
      }
    }

    function closeTour() {
      if (tourOverlay) tourOverlay.style.display = "none";
      try { localStorage.setItem("divya_tour_completed", "true"); } catch (e) {}
    }

    if (btnTourClose) btnTourClose.addEventListener("click", closeTour);
    if (btnTourPrev) {
      btnTourPrev.addEventListener("click", () => {
        if (currentTourStep > 0) {
          currentTourStep--;
          updateTourCard();
        }
      });
    }
    if (btnTourNext) {
      btnTourNext.addEventListener("click", () => {
        if (currentTourStep < TOUR_STEPS.length - 1) {
          currentTourStep++;
          updateTourCard();
        } else {
          closeTour();
        }
      });
    }
    if (btnRestartTour) {
      btnRestartTour.addEventListener("click", () => {
        const sd = document.getElementById("settings-drawer");
        if (sd) sd.classList.remove("open");
        startTour();
      });
    }

    const btnCheckUpdate = document.getElementById("btn-check-update");
    const updateStatus = document.getElementById("update-status");
    if (btnCheckUpdate) {
      btnCheckUpdate.addEventListener("click", async () => {
        playAudioTone(880, "triangle", 0.08, 0.2);
        if (updateStatus) updateStatus.textContent = "Checking network...";
        badge("CHECKING FOR UPDATES...");

        if ("serviceWorker" in navigator) {
          try {
            const reg = await navigator.serviceWorker.getRegistration();
            if (reg) {
              await reg.update();
              if (reg.waiting) {
                if (updateStatus) updateStatus.textContent = "Reloading new version...";
                badge("NEW UPDATE READY · RELOADING");
                reg.waiting.postMessage({ type: "SKIP_WAITING" });
                setTimeout(() => window.location.reload(), 600);
                return;
              }
            }
          } catch (e) {
            console.warn("SW update check error:", e);
          }
        }

        try {
          const res = await fetch("./index.html?nocache=" + Date.now(), { cache: "no-store" });
          if (res.ok) {
            if (updateStatus) updateStatus.textContent = "Up to date · Git Synced";
            badge("APP IS UP TO DATE");
          } else {
            if (updateStatus) updateStatus.textContent = "Offline Mode (Cached)";
            badge("OFFLINE MODE · CACHED");
          }
        } catch (e) {
          if (updateStatus) updateStatus.textContent = "Offline Mode (Cached)";
          badge("OFFLINE READY");
        }
      });
    }

    // Auto-launch on first visit after entering studio
    const origEnterBtn = document.getElementById("btn-enter");
    if (origEnterBtn) {
      origEnterBtn.addEventListener("click", () => {
        let done = false;
        try { done = localStorage.getItem("divya_tour_completed") === "true"; } catch (e) {}
        if (!done) {
          try { localStorage.setItem("divya_tour_completed", "true"); } catch (e) {}
          setTimeout(() => startTour(), 600);
        }
      });
    }

    window.addEventListener("resize", () => {
      if (tourOverlay && tourOverlay.style.display !== "none") {
        updateTourCard();
      }
    });

    if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
      navigator.serviceWorker.register("./sw.js").catch(() => {});
    }

    // Startup
    window.addEventListener("DOMContentLoaded", () => {
      fitViewport();
      initVisionEngine();
      requestAnimationFrame(renderLoop);
    });
