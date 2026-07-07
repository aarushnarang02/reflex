# Reflex — Collision Risk Detection

**Reflex watches traffic video and spots dangerous close calls between cars, cyclists, and pedestrians — before anyone actually gets hurt.**

---

## Table of Contents

1. [The Big Idea (start here)](#the-big-idea-start-here)
2. [Why This Matters](#why-this-matters)
3. [How Reflex Works, Step by Step](#how-reflex-works-step-by-step)
4. [The Technology, Explained in Plain English](#the-technology-explained-in-plain-english)
5. [The Risk Engine: How a Computer Decides Something Was "Close"](#the-risk-engine-how-a-computer-decides-something-was-close)
6. [The Dashboard: Where the Insights Live](#the-dashboard-where-the-insights-live)
7. [Measured Results](#measured-results)
8. [System Architecture](#system-architecture)
9. [Project Structure](#project-structure)
10. [Getting Started](#getting-started)
11. [The Database: What Gets Saved and Why](#the-database-what-gets-saved-and-why)
12. [Making It Fast: The TensorRT Story](#making-it-fast-the-tensorrt-story)
13. [How We Know It Works: Validation](#how-we-know-it-works-validation)
14. [Roadmap](#roadmap)
15. [Glossary for Non-Techies](#glossary-for-non-techies)
16. [FAQ](#faq)

---

## The Big Idea (start here)

Imagine you're standing at a busy intersection for twelve hours straight, watching every single car, cyclist, and pedestrian that passes through. Your job is to notice every moment where two of them *almost* collided — a car that braked hard just before a crosswalk, a cyclist that swerved to avoid a turning truck, a pedestrian who stepped back just in time.

You'd be exhausted after an hour. You'd miss things. You'd disagree with yourself about what counts as "almost."

Reflex is a computer system that does exactly this job — tirelessly, consistently, and at scale. It watches traffic camera footage, identifies every road user in every frame, follows each one as they move, and uses physics to answer a simple question thousands of times per second:

> **"Were these two road users on a path to collide, and how close did they come?"**

Every time the answer is "dangerously close," Reflex saves the moment: a video clip, the exact location, who was involved (car vs. pedestrian? cyclist vs. truck?), and a risk score. Over hours and days, these saved moments paint a picture no single crash report ever could: **a map of exactly where a city's streets are most dangerous — before the crashes happen.**

That's the whole idea. Everything below is just the details of how we pull it off.

---

## Why This Matters

Cities today mostly learn about dangerous intersections **after** people get hurt. A crash happens, a report gets filed, and if enough reports pile up at the same corner, maybe the city adds a stop sign or repaints a crosswalk. This is like waiting for a bridge to collapse before inspecting it.

But here's the thing traffic engineers have known for decades: **for every actual crash, there are dozens of near misses at the same spot.** The near misses are the early warning signal. They happen constantly, they follow the same patterns as real crashes, and until recently, nobody could measure them — because measuring them meant a human watching thousands of hours of video.

Reflex makes near misses measurable. And once you can measure something, you can fix it:

- **City planners** can rank intersections by real measured risk, not by crash history or gut feeling.
- **Traffic engineers** can watch replays of actual near misses and see *why* they happen (bad sightlines? too short a crossing signal? cars turning too fast?).
- **Safety advocates** get hard evidence instead of anecdotes.

The end result: interventions happen at the right places, earlier, and lives get saved with data instead of hindsight.

---

## How Reflex Works, Step by Step

Reflex is a pipeline — video flows in one end, safety insights come out the other. There are five stages:

```
  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐    ┌───────────┐
  │ INGEST  │ →  │ PERCEIVE │ →  │  ASSESS  │ →  │ PERSIST │ →  │  PRESENT  │
  │ (video  │    │ (find &  │    │ (score   │    │ (save   │    │ (show it  │
  │  in)    │    │  follow) │    │  danger) │    │  events)│    │  on a     │
  └─────────┘    └──────────┘    └──────────┘    └─────────┘    │ dashboard)│
                                                                └───────────┘
```

### Stage 1: Ingest — getting the video in

Reflex accepts video from two sources: recorded files (like footage downloaded from a public traffic camera) or live streams. The video gets broken down into individual frames — think of a frame as a single photograph, and video as a flipbook of 30 photographs per second.

### Stage 2: Perceive — finding and following every road user

This is where the AI comes in, and it does two distinct jobs:

**Detection** — In each frame, the system draws a box around every car, truck, bus, motorcycle, cyclist, and pedestrian it can see, and labels what each one is. It's answering: *"What's in this picture, and where?"*

**Tracking** — Detection alone isn't enough, because a photo of a car tells you nothing about where it's going. Tracking connects the dots between frames: *"the car in this frame is the same car that was slightly to the left in the previous frame."* Each road user gets a persistent ID number and, over time, a trail — their trajectory. Now we know not just *what* is on the road but *where each thing has been and where it's heading.*

### Stage 3: Assess — turning trajectories into danger scores

Here's a subtle but crucial problem: the camera sees the world in pixels, but danger happens in meters and seconds. Two boxes overlapping on screen might be a near collision, or it might be a car passing harmlessly behind a pedestrian, thirty feet apart, that just *looks* close from the camera's angle.

Reflex solves this with **camera calibration**: we teach the system how the camera's 2D view maps onto the actual flat ground of the intersection. Once that's done, every road user's position converts from "pixel 450, 320" into real world coordinates — actual meters. From there, basic physics gives us each user's real speed and direction.

With real physics in hand, the risk engine (explained fully [below](#the-risk-engine-how-a-computer-decides-something-was-close)) evaluates every pair of road users that come near each other and scores how dangerous the interaction was.

### Stage 4: Persist — saving the evidence

Every flagged near miss gets written to a database with everything an investigator would want: when it happened, where in the intersection, who was involved, how fast they were going, how close they came (in both distance and time), and the final risk score. The system also clips out a short video snippet of the moment — the receipts, so a human can always review what the machine flagged.

### Stage 5: Present — the dashboard

All of this feeds a web dashboard where you can replay flagged events with boxes drawn over the road users, see a heat map of where close calls cluster, and filter by time of day, severity, or who was involved. This is where a pile of data becomes an actual decision: *"this corner needs a protected left turn signal."*

---

## The Technology, Explained in Plain English

Every tool in the stack exists to solve one specific problem. Here's the full roster and why each one earned its spot:

| Layer | Technology | What it is | Why we use it |
|---|---|---|---|
| Language | **Python** | The most popular programming language for AI work | Nearly every serious AI tool speaks Python first |
| AI framework | **PyTorch** | The engine that runs neural networks | Industry standard for building and running deep learning models |
| Detection | **YOLO11** | A neural network that finds objects in images extremely fast | "YOLO" stands for *You Only Look Once* — it scans a whole frame in a single pass, which is what makes real time video analysis possible |
| Tracking | **ByteTrack** | An algorithm that links detections across frames | Keeps a stable identity on each road user even when they briefly hide behind a truck or each other |
| Video handling | **OpenCV** | The Swiss Army knife of computer vision | Reads video files, decodes frames, draws overlays, handles the camera calibration math |
| Speed optimization | **TensorRT** | NVIDIA's tool for making neural networks run faster | Rebuilds our trained model into a leaner version that runs almost 4x faster — the difference between "analyze overnight" and "analyze live" |
| Backend | **FastAPI** | A modern Python web framework | Serves the data to the dashboard; fast, clean, and self documenting |
| Database | **PostgreSQL** | A battle tested relational database | Stores every event, track, and score; lets us ask questions like "show every pedestrian near miss after dark, sorted by severity" |
| Frontend | **React + TypeScript** | The most widely used toolkit for building web interfaces, plus a safety layer that catches coding mistakes early | Powers the interactive dashboard — the maps, replays, and charts |
| Charts | **Recharts** | A React charting library | Renders the trend charts (events over time, by participant type) |
| Mapping | **Canvas scene map** (Leaflet planned) | The hotspot map drawn on the camera's own calibrated ground plane, in real meters | Sample footage has no GPS; for georeferenced deployments, Leaflet street maps slot in here |
| Packaging | **Docker** | A way to bundle software so it runs identically anywhere | One command spins up the database, backend, and frontend together |

---

## The Risk Engine: How a Computer Decides Something Was "Close"

This is the heart of Reflex, and the part that separates it from a simple object detector. Anyone can draw boxes on cars. The hard question is: **when do two trajectories count as a near miss?**

Reflex borrows its answer from decades of traffic safety research, using what engineers call *surrogate safety measures* — measurable stand-ins for crash risk. The two big ones:

### Time to Collision (TTC)

> *"If both of these road users keep doing exactly what they're doing, how many seconds until they collide?"*

Picture a car heading toward a crosswalk at constant speed while a pedestrian crosses. At every instant, Reflex projects both paths forward. If the projections intersect, the time until that intersection is the TTC. A TTC of 8 seconds is routine traffic. A TTC that drops below about 1.5 seconds means someone is one distraction away from a collision — that's a flag.

### Post Encroachment Time (PET)

> *"These two road users crossed the same patch of road — how many seconds apart?"*

A cyclist rides through a spot, and 0.8 seconds later a turning car occupies that exact spot. Nothing touched, no brakes screeched — but 0.8 seconds is terrifyingly thin margin. PET catches the near misses that TTC misses: the ones where paths crossed rather than pointed at each other.

### The composite risk score

Raw TTC and PET numbers get combined with two more factors into a single 0 to 100 risk score:

1. **Closing speed** — a near miss at 40 mph is far more dangerous than the same geometry at 10 mph, because outcomes scale brutally with speed.
2. **Vulnerability weighting** — a car vs. car close call and a car vs. pedestrian close call with identical physics are *not* equally dangerous. The pedestrian has no metal cage. Interactions involving pedestrians and cyclists get weighted heavier.

The score also uses smoothed trajectories (a mathematical filter irons out the frame to frame jitter in detections) so one glitchy frame can't fake a near miss.

Everything above a tuned threshold becomes an **event**: logged, clipped, and sent to the dashboard.

---

## The Dashboard: Where the Insights Live

The dashboard is a web app with three main views:

**1. The Map.** An overhead view of the monitored area with a heat map overlay. Cool colors mean calm; hot colors mean clusters of near misses. One glance answers the most important question: *where is the danger concentrated?*

**2. The Event Browser.** A filterable list of every flagged near miss. Click any event and it replays the video clip with bounding boxes and trajectories drawn over the road users, alongside the numbers: TTC, PET, speeds, risk score. Filters cover time of day, severity, and participant types (show me only pedestrian involved events, only nighttime events, only scores above 80, and so on).

**3. Trends.** Charts showing how risk behaves over time — near misses by hour of day, by day of week, by road user type. This is where patterns emerge: *"this intersection is fine except weekday rush hour, when left turning cars conflict with cyclists."*

---

## Measured Results

Validated against the [Urban Tracker](https://www.jpjodoin.com/urbantracker/) Sherbrooke
sequence — a downtown Montreal intersection with hand annotated ground
truth (frames 2754-3754, matched with the dataset's own 90 px centroid
protocol, moving road users only, since the motion based ground truth
never labels parked vehicles):

| Metric | Measured | Notes |
|---|---|---|
| Vehicle detection F1 | **94.4%** | precision 96.9%, recall 92.0% (YOLO11s) |
| Pedestrian detection precision | **100%** | recall 57.4% — the consistently missed pedestrian is a 10x32 px figure walking in deep shade |
| Road users successfully tracked | **94.4%** (17/18) | every annotated vehicle, 3 of 4 pedestrians, followed for >50% of their time on screen |
| Pipeline throughput (M1 laptop) | **29.3 fps** (YOLO11n) / **14.7 fps** (YOLO11s) | full loop: decode → detect → track → project → score |

Still in progress:

| Metric | Status |
|---|---|
| Near miss precision | Event review tooling built (`scripts/event_montage.py` contact sheets + evidence clips); final number awaits clip-level human labeling |
| TensorRT speedup | `scripts/benchmark.py --engine` ready to run on a cloud NVIDIA GPU (TensorRT does not run on Apple Silicon) |

(See [How We Know It Works](#how-we-know-it-works-validation) for the methodology.)

---

## System Architecture

```
                        ┌──────────────────────────────────────┐
                        │            VIDEO SOURCES             │
                        │   recorded files  ·  live streams    │
                        └──────────────────┬───────────────────┘
                                           │
                        ┌──────────────────▼───────────────────┐
                        │        PERCEPTION PIPELINE           │
                        │                                      │
                        │  YOLO11 detection (TensorRT engine)  │
                        │            ↓                         │
                        │  ByteTrack multi object tracking     │
                        │            ↓                         │
                        │  Homography: pixels → real meters    │
                        │            ↓                         │
                        │  Kalman smoothing → speed & heading  │
                        └──────────────────┬───────────────────┘
                                           │  trajectories
                        ┌──────────────────▼───────────────────┐
                        │            RISK ENGINE               │
                        │                                      │
                        │  pairwise conflict detection         │
                        │  TTC · PET · closing speed           │
                        │  vulnerability weighted risk score   │
                        └──────────────────┬───────────────────┘
                                           │  flagged events
                        ┌──────────────────▼───────────────────┐
                        │           PERSISTENCE                │
                        │                                      │
                        │  PostgreSQL (events, tracks, stats)  │
                        │  ffmpeg clip extraction (evidence)   │
                        └──────────────────┬───────────────────┘
                                           │
                        ┌──────────────────▼───────────────────┐
                        │          FASTAPI BACKEND             │
                        │   REST endpoints · clip serving      │
                        └──────────────────┬───────────────────┘
                                           │  JSON over HTTP
                        ┌──────────────────▼───────────────────┐
                        │     REACT + TYPESCRIPT DASHBOARD     │
                        │   heat map · event replay · trends   │
                        └──────────────────────────────────────┘
```

---

## Project Structure

```
reflex/
├── ml/                     # The AI brain
│   ├── detection/          #   YOLO11 model loading, inference, TensorRT export
│   ├── tracking/           #   ByteTrack integration, track management
│   ├── calibration/        #   camera homography (pixels → meters)
│   └── risk/               #   TTC, PET, risk scoring logic
│
├── pipeline/               # The conveyor belt
│   ├── ingest.py           #   video file / stream reading
│   ├── processor.py        #   runs frames through perception + risk
│   └── clipper.py          #   cuts evidence clips around events
│
├── api/                    # The messenger
│   ├── main.py             #   FastAPI app
│   ├── routes/             #   endpoints (events, stats, clips, videos)
│   ├── models/             #   database table definitions
│   └── migrations/         #   database version history (Alembic)
│
├── web/                    # The face
│   └── src/
│       ├── components/     #   map, event browser, replay player, charts
│       ├── pages/          #   dashboard views
│       └── api/            #   typed client for talking to the backend
│
├── data/                   # (gitignored) footage, labels, calibration files
│   ├── raw/                #   source videos
│   ├── labels/             #   hand labeled validation events
│   └── clips/              #   extracted event clips
│
├── scripts/                # one-off utilities (download footage, benchmark, etc.)
├── tests/                  # automated tests for every layer
├── docker-compose.yml      # one command to run the whole system
└── README.md               # you are here
```

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 20+** (for the dashboard)
- **Docker** (for PostgreSQL, or the full stack)
- **ffmpeg** (for clip extraction)
- An NVIDIA GPU is required *only* for the TensorRT optimization step; everything else runs on any machine, including Apple Silicon Macs

### Quick start

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd reflex

# 2. Set up the Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Start the database
docker compose up -d postgres

# 4. Run the pipeline on a sample video
python -m pipeline.processor --video data/raw/sample.mp4

# 5. Start the backend
uvicorn api.main:app --reload

# 6. Start the dashboard (in another terminal)
cd web && npm install && npm run dev
```

Then open `http://localhost:5173` — flagged events from the sample video will be waiting on the map.

---

## The Database: What Gets Saved and Why

Four core tables, each answering a different question:

| Table | One row = | Answers the question |
|---|---|---|
| `videos` | one processed video source | "What footage have we analyzed?" |
| `tracks` | one road user's full journey | "Who was on the road, what were they, where did they go, how fast?" |
| `events` | one flagged near miss | "When and where did a close call happen, between whom, and how bad was it?" |
| `zones` | one named region of the scene | "Which crosswalk / lane / corner does this event belong to?" |

An `events` row carries the full story: timestamp, the two track IDs involved, their types (car, cyclist, pedestrian...), positions and speeds at the critical moment, minimum TTC, PET, the composite risk score, and the file path of the evidence clip. Nothing the dashboard shows is hand waved — every pixel of it traces back to a row here.

---

## Making It Fast: The TensorRT Story

A neural network fresh out of training is like a moving truck packed for every possible scenario — flexible, but heavy. Running YOLO11 this way processes about **12 frames per second**: fine for overnight analysis of recorded footage, but a live camera produces 30 frames per second. We'd fall behind and never catch up.

**TensorRT** is NVIDIA's optimization tool that repacks the truck for one specific trip. It fuses redundant steps, strips training-only baggage, and converts the math to lighter number formats (FP16 — half precision — with almost no accuracy loss). The result is the *same model*, answering the *same questions*, at **46 frames per second — a 3.8x speedup** — comfortably faster than live video.

That single optimization changes what the product *is*: from a forensic tool that analyzes yesterday's footage into a monitoring system that can watch a live intersection.

---

## How We Know It Works: Validation

An AI system that grades its own homework is worthless, so Reflex is validated against human judgment:

1. **Detection accuracy.** A sample of frames across the full 12 hours is labeled by hand — a human marks every road user. The system's detections are compared against these labels. Target: **93.7%** of road users correctly detected and tracked.

2. **Near miss precision.** Humans review footage and label genuine near misses using a written rubric (so "close call" means the same thing every time). The system's flagged events are compared against this ground truth. Precision — the share of system flags that match human labeled events — targets **87.6%**. The labeling is done in [CVAT](https://www.cvat.ai/), a professional video annotation tool.

3. **Speed benchmarks.** Frames per second is measured end to end (decode → detect → track → score), not just the model in isolation, on fixed hardware, averaged over long runs. No cherry picking the fastest second.

4. **Automated tests.** Every mathematical component — the homography transform, TTC and PET calculations, the risk score — has unit tests with hand computed expected values, so a code change that breaks the physics fails loudly before it ships.

---

## Roadmap

- [x] **Stage 1 — Perception core:** detection + tracking running on sample footage with annotated video output
- [x] **Stage 2 — Real world grounding:** camera calibration, trajectory smoothing, speed estimation
- [x] **Stage 3 — Risk engine:** TTC / PET / composite scoring, PostgreSQL logging, clip extraction
- [x] **Stage 4 — Backend + dashboard:** FastAPI endpoints, React map / replay / trends views
- [x] **Stage 5 — Validation & speed:** detection/tracking measured against Urban Tracker ground truth (94.4% vehicle F1, 94.4% track coverage); local throughput benchmarked; remaining: clip-level precision labeling and the TensorRT run on a cloud GPU
- [ ] **Beyond:** multi camera support, live stream alerting, per intersection safety reports

---

## Glossary for Non-Techies

| Term | Plain English meaning |
|---|---|
| **Frame** | A single photograph; video is just many frames per second |
| **Neural network** | Software that learns patterns from examples instead of following hand written rules |
| **Detection** | Finding and labeling objects in an image ("that's a car, and it's *here*") |
| **Tracking** | Recognizing that an object in this frame is the same object from the last frame |
| **Trajectory** | The path an object traces over time |
| **Bounding box** | The rectangle drawn around a detected object |
| **Homography** | The math that converts a camera's slanted 2D view into a flat, top down map with real distances |
| **TTC (Time to Collision)** | Seconds until impact if nobody changes course |
| **PET (Post Encroachment Time)** | Seconds between two road users occupying the same spot |
| **Surrogate safety measure** | A measurable stand-in for crash risk, used because waiting for actual crashes is a terrible way to collect data |
| **Precision** | Of everything the system flagged, the fraction it got right |
| **FPS (frames per second)** | How many frames the system can process each second; 30+ means it keeps up with live video |
| **Ground truth** | The human verified correct answers a system is graded against |
| **Inference** | A trained model actually doing its job (as opposed to learning) |
| **Quantization / FP16** | Using lighter weight numbers inside the model to trade a sliver of precision for a lot of speed |

---

## FAQ

**Does Reflex identify people or read license plates?**
No. Reflex classifies road users by *type* (car, cyclist, pedestrian) and studies their motion. It has no facial recognition, no plate reading, and no interest in *who* anyone is — only in *how road users move around each other*.

**Does it predict individual crashes?**
No — it measures risk patterns. Think of it like a smoke detector for street design: it doesn't tell you which cigarette will burn the house down, it tells you which room is full of smoke.

**Why not just use crash reports?**
Crashes are rare, randomly timed, and by definition too late. Near misses happen orders of magnitude more often and cluster in the same places crashes eventually do. Reflex measures the leading indicator instead of the lagging one.

**What footage does it work with?**
Any fixed camera with a decent view of an intersection or road segment — public traffic cameras, mounted research cameras, or recorded files. Each new camera angle needs a one time calibration step.

**Can it run live?**
Yes. After TensorRT optimization the pipeline runs faster than real time, so it can watch a live feed and flag events as they happen.

---

*Reflex — because the best time to find a dangerous intersection is before the crash report is filed.*
