# Software + Physics Project Ideas

Created: 2026-01-28

## Context
- Student needing internship by April
- Interests: astronomy, quantum computing, light speed travel, wormholes, parallel realities
- Constraint: IoT is expensive (but ESP32 is cheap)
- Goal: Build something meaningful, not another generic app

---

## Tier 1: High Impact, Actually Buildable

### 1. Exoplanet Transit Detector
- Use real NASA TESS/Kepler data (free, public)
- Build algorithm to detect brightness dips = planet crossing star
- Visualize the light curves
- **Why it's good**: Real science, real data, shows data analysis skills
- **Resources**: NASA Exoplanet Archive, Lightkurve Python library

### 2. Relativistic Flight Simulator
- Visualize what you'd actually SEE at 0.9c
- Doppler shift, aberration, length contraction
- WebGL/Three.js for rendering
- **Why it's good**: Physics + graphics + math. Interviewers remember this.

### 3. N-Body Gravity Sandbox
- Simulate solar systems, galaxies, chaotic 3-body problems
- Real physics (Verlet integration, Barnes-Hut for performance)
- Web-based, shareable
- **Why it's good**: Demonstrates algorithms, physics understanding, visualization

### 4. Quantum Circuit Visualizer
- Visual drag-drop quantum gate builder
- Show state evolution on Bloch sphere
- Connect to IBM Qiskit for real hardware execution
- **Why it's good**: Quantum is hot, educational tools are needed, shows you understand the math
- **Resources**: IBM Qiskit (free), Google Cirq

---

## Tier 2: IoT on a Budget

ESP32 boards are $3-5. Sensors are cheap.

### 5. Light Pollution Monitor Network
- TSL2591 light sensor (~$6) + ESP32
- Log sky brightness over time
- Map data from multiple locations
- Useful for amateur astronomers

### 6. Magnetometer for Aurora Alerts
- Detect geomagnetic disturbances
- Push notifications when aurora likely
- Actual utility for people in northern latitudes

---

## Tier 3: Weird But Memorable

### 7. Spacetime Curvature in AR
- Phone AR showing how mass warps space
- Drop objects, watch geodesics curve
- Nobody has done this well

### 8. Pulsar Sound Synthesizer
- Convert real pulsar timing data to audio
- Each pulsar = unique rhythm/tone
- Art + science crossover

---

## Internship Targeting

| Target Field | Best Projects |
|--------------|---------------|
| Aerospace/Space Tech | 1, 3, 5 |
| Quantum Computing | 4 |
| General SWE (stand out) | 2, 7 |
| Research Labs | 1 (paper potential) |

---

## 2026 Trends (from research)

- **Clawdbot/Moltbot**: Personal AI assistant by Peter Steinberger, 30K+ GitHub stars
- **Local-first AI**: Privacy-preserving, runs on your hardware
- **AI with hands**: Not just chat, actually does things
- **Superpowers/skills systems**: Extending Claude Code with reusable capabilities
- **Niche vertical tools**: Specific industries beat generic solutions

---

## Underserved Niches (if pivoting to practical)

1. Small landlords (<10 units) - tracking, expenses, tenant comms
2. HVAC/plumber/electrician contractors - scheduling, routes, communication
3. Small SaaS compliance - SOC2 for startups without $20K/year tools
4. Nonprofits - donor tracking, grant applications

---

## Next Steps

1. Pick ONE project from Tier 1
2. Build MVP in 2-3 weeks
3. Document the physics/math behind it
4. Deploy publicly (GitHub + live demo)
5. Write about it (blog post or README that explains the science)
6. Add to resume/portfolio before internship applications

---

## Notes

The goal isn't GitHub stars. It's building something YOU actually care about.
Peter Steinberger built Clawdbot for himself first. Stars came after.
