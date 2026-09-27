"""
High-End Synchronized Broadcast Player for Continuous Real-Time Streaming.
Operates natively on port 8501 via Streamlit static media serving.
Provides 60 FPS continuous video playback, real-time scorecard overlays,
bottom-stacked scrollable commentary feed, and spoken audio narration.
"""

import json
from typing import Any


def render_synchronized_broadcast_player(
    stream_deliveries_payload: list[dict[str, Any]],
    initial_score: str = "190/5",
    initial_overs: str = "18.0",
    initial_striker: str = "LS Livingstone",
    initial_bowler: str = "HV Patel",
    initial_crr: str = "10.56",
) -> str:
    """
    Renders a unified client-side broadcast dashboard that synchronizes:
    1. Top Match Header Scoreboard Banner
    2. 60 FPS Continuous Video Stream (with dynamic origin resolution)
    3. On-screen TV Overlay Score Ribbon
    4. Control Deck (Play/Pause, Next Delivery, Reset) with atomic event dispatch
    5. Bottom-stacked auto-scrolling commentary cards
    6. Parallel spoken voice audio playback & pause handling
    """
    payload_json = json.dumps(stream_deliveries_payload)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #F0F4F8;
            overflow: hidden;
        }}
        
        /* Unified Header Scoreboard Banner */
        .hdr-banner {{
            background: linear-gradient(135deg, rgba(13, 22, 48, 0.95) 0%, rgba(20, 32, 70, 0.95) 100%);
            border: 1px solid rgba(0, 240, 255, 0.35);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }}
        .hdr-live-badge {{
            background: #FF0055;
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
            display: inline-block;
            transition: all 0.3s ease;
        }}
        .hdr-live-badge.paused {{
            background: #718096;
        }}
        .hdr-match-title {{
            font-weight: 700;
            color: #E2E8F0;
            font-size: 1.05rem;
        }}
        .hdr-sub {{
            font-size: 0.85rem;
            color: #A0AEC0;
            margin-top: 3px;
        }}
        .hdr-score-val {{
            font-size: 1.65rem;
            font-weight: 800;
            color: #00F0FF;
            text-align: right;
            line-height: 1.1;
        }}
        .hdr-score-ov {{
            font-size: 0.95rem;
            color: #A0AEC0;
            font-weight: 500;
        }}
        .hdr-crr {{
            color: #00E5FF;
            font-weight: 600;
            font-size: 0.82rem;
            text-align: right;
            margin-top: 2px;
        }}

        /* Player Card */
        .player-card {{
            background: rgba(10, 17, 40, 0.95);
            border: 1px solid rgba(0, 240, 255, 0.25);
            border-radius: 12px;
            padding: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
        }}
        .video-container {{
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            margin-bottom: 10px;
        }}
        video {{
            width: 100%;
            height: 330px;
            display: block;
            background: #000;
            object-fit: cover;
        }}
        .tv-overlay {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            background: rgba(7, 11, 25, 0.9);
            border-left: 4px solid #00F0FF;
            border-radius: 6px;
            padding: 8px 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.6);
            z-index: 10;
        }}
        .btn-deck {{
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }}
        .btn {{
            flex: 1;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: 700;
            cursor: pointer;
            font-size: 0.88rem;
            transition: all 0.2s ease;
            text-align: center;
            border: none;
            outline: none;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #00F0FF 0%, #0088FF 100%);
            color: #070B19;
        }}
        .btn-primary:hover {{
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.6);
            transform: translateY(-1px);
        }}
        .btn-secondary {{
            background: rgba(0, 240, 255, 0.12);
            color: #00F0FF;
            border: 1px solid rgba(0, 240, 255, 0.4);
        }}
        .btn-secondary:hover {{
            background: rgba(0, 240, 255, 0.25);
        }}
        .btn-reset {{
            background: rgba(255, 255, 255, 0.08);
            color: #CBD5E0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .btn-reset:hover {{
            background: rgba(255, 255, 255, 0.15);
        }}
        .comm-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 6px;
        }}
        .feed-box {{
            max-height: 200px;
            height: 200px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
            scroll-behavior: smooth;
            padding-right: 4px;
        }}
        .feed-box::-webkit-scrollbar {{
            width: 5px;
        }}
        .feed-box::-webkit-scrollbar-thumb {{
            background: rgba(0, 240, 255, 0.4);
            border-radius: 4px;
        }}
        .feed-box::-webkit-scrollbar-track {{
            background: rgba(0, 0, 0, 0.2);
        }}
        .card-lead {{
            background: rgba(0, 229, 255, 0.08);
            border-left: 4px solid #00F0FF;
            border-radius: 6px;
            padding: 8px 10px;
            animation: fadeIn 0.3s ease;
        }}
        .card-analyst {{
            background: rgba(255, 184, 0, 0.08);
            border-left: 4px solid #FFB800;
            border-radius: 6px;
            padding: 8px 10px;
            animation: fadeIn 0.3s ease;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    </head>
    <body>

    <!-- 1. ATOMIC UNIFIED MATCH HEADER SCOREBOARD -->
    <div class="hdr-banner">
        <div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:3px;">
                <span id="hdr-badge" class="hdr-live-badge">● LIVE BROADCAST</span>
                <span class="hdr-match-title">3rd T20I: England vs India</span>
                <span style="color:#718096; font-size:0.8rem;">Trent Bridge</span>
            </div>
            <div class="hdr-sub">
                Batting: <strong style="color:white;">England</strong> | 
                Striker: <strong id="hdr-striker" style="color:#00F0FF;">{initial_striker}</strong> | 
                Bowler: <strong id="hdr-bowler" style="color:#FFB800;">{initial_bowler}</strong>
            </div>
        </div>
        <div>
            <div class="hdr-score-val">
                <span id="hdr-score">{initial_score}</span> <span id="hdr-overs" class="hdr-score-ov">({initial_overs} ov)</span>
            </div>
            <div id="hdr-crr" class="hdr-crr">CRR: {initial_crr}</div>
        </div>
    </div>

    <!-- 2. BROADCAST STAGE CARD -->
    <div class="player-card">
        <!-- Live Video Element -->
        <div class="video-container">
            <video id="sync-match-video" controls playsinline preload="auto">
                <source src="/app/static/match.mp4" type="video/mp4">
                <source src="static/match.mp4" type="video/mp4">
                Your browser does not support HTML5 video streaming.
            </video>
            
            <!-- Real-Time On-Screen TV Score Ribbon Overlay -->
            <div class="tv-overlay" id="tv-score-overlay">
                <div>
                    <div style="font-weight: 800; color: #00F0FF; font-size: 1.05rem;" id="overlay-score">ENG {initial_score} ({initial_overs} ov)</div>
                    <div style="font-size: 0.78rem; color: #E2E8F0;" id="overlay-players">🏏 {initial_striker} | 🎯 {initial_bowler}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; color: #FFD700; font-weight: 700;" id="overlay-last-ball">READY (24.0s)</div>
                    <div id="sync-status" style="font-size: 0.72rem; color: #718096;">● Over 18 Ready</div>
                </div>
            </div>
        </div>

        <!-- Broadcast Control Deck -->
        <div class="btn-deck">
            <button id="btn-play-live" class="btn btn-primary" onclick="togglePlayLiveStream()">
                ▶ Play Live Stream
            </button>
            <button id="btn-next-delivery" class="btn btn-secondary" onclick="stepNextDelivery()">
                ⏭ Next Delivery
            </button>
            <button id="btn-reset-over18" class="btn btn-reset" onclick="resetToOver18()">
                🔄 Reset (Over 18.0)
            </button>
        </div>

        <!-- Live Commentary Bottom-Stack Container -->
        <div class="comm-section">
            <div class="comm-header">
                <span style="font-size: 0.82rem; font-weight: 800; color: #00F0FF; letter-spacing: 0.8px;">🎙️ REAL-TIME AI COMMENTARY (JAMES & HARSHA)</span>
                <span id="audio-indicator" style="font-size: 0.72rem; color: #00E5FF;">🔊 Voice Ready</span>
            </div>
            
            <div id="live-commentary-feed" class="feed-box">
                <div style="background: rgba(0, 229, 255, 0.05); border-left: 3px solid #00F0FF; border-radius: 6px; padding: 8px 10px; font-size: 0.82rem; color: #A0AEC0;">
                    Broadcast ready at Over 18.0 (24.0s). Click <strong>Play Live Stream</strong> or <strong>Next Delivery</strong> to begin continuous commentary.
                </div>
            </div>
        </div>

        <!-- Audio Element for Instant Voice Playback -->
        <audio id="commentary-voice-player" preload="auto" style="display: none;"></audio>
    </div>

    <script>
        const deliveries = {payload_json};
        let currentDeliveryIdx = 0;
        let isLivePlaying = false;
        const video = document.getElementById('sync-match-video');
        const audio = document.getElementById('commentary-voice-player');
        const commFeed = document.getElementById('live-commentary-feed');
        
        // Scorecard elements (Header & Overlay)
        const hdrBadge = document.getElementById('hdr-badge');
        const hdrScore = document.getElementById('hdr-score');
        const hdrOvers = document.getElementById('hdr-overs');
        const hdrStriker = document.getElementById('hdr-striker');
        const hdrBowler = document.getElementById('hdr-bowler');
        const hdrCrr = document.getElementById('hdr-crr');
        
        const scoreOverlay = document.getElementById('overlay-score');
        const playersOverlay = document.getElementById('overlay-players');
        const lastBallOverlay = document.getElementById('overlay-last-ball');
        const syncStatus = document.getElementById('sync-status');
        const playBtn = document.getElementById('btn-play-live');
        const audioIndicator = document.getElementById('audio-indicator');

        // Set of triggered ball indices to avoid duplicate triggers
        let triggeredDeliveries = new Set();

        // Safe origin resolution for iframe contexts
        function getBaseOrigin() {{
            let origin = "";
            try {{
                if (window.parent && window.parent.location && window.parent.location.origin) {{
                    origin = window.parent.location.origin;
                }}
            }} catch (e) {{}}
            if (!origin || origin === "null" || origin.startsWith("about")) {{
                origin = window.location.origin;
            }}
            if (!origin || origin === "null" || origin.startsWith("about")) {{
                origin = window.location.protocol + "//" + (window.location.host || "localhost:8501");
            }}
            return origin;
        }}

        const baseOrigin = getBaseOrigin();

        // Initialize Video Source with dynamic absolute URL
        function initVideoSource() {{
            const absVideoUrl = baseOrigin + "/app/static/match.mp4";
            if (!video.src || video.src.startsWith("about")) {{
                video.src = absVideoUrl;
                video.load();
            }}
            video.currentTime = 24.0;
        }}

        initVideoSource();

        video.addEventListener('loadedmetadata', function() {{
            if (video.currentTime < 24.0) {{
                video.currentTime = 24.0;
            }}
        }});

        // ATOMIC EVENT DISPATCHER
        function triggerDeliveryEvent(d, idx) {{
            if (triggeredDeliveries.has(idx)) return;
            triggeredDeliveries.add(idx);
            currentDeliveryIdx = idx + 1;

            // 1. Atomically Update Main Top Header Scorecard
            if (hdrScore) hdrScore.innerText = d.score;
            if (hdrOvers) hdrOvers.innerText = `(${{d.overs_display}} ov)`;
            if (hdrStriker) hdrStriker.innerText = d.batter;
            if (hdrBowler) hdrBowler.innerText = d.bowler;
            if (hdrCrr) hdrCrr.innerText = d.crr ? `CRR: ${{d.crr}}` : 'CRR: 10.56';

            // 2. Atomically Update On-Screen TV Score Ribbon
            scoreOverlay.innerText = `${{d.team}} ${{d.score}} (${{d.overs_display}} ov)`;
            playersOverlay.innerText = `🏏 ${{d.batter}} | 🎯 ${{d.bowler}}`;
            lastBallOverlay.innerText = `Over ${{d.over}}.${{d.ball}}: ${{d.runs_total}} run${{d.runs_total !== 1 ? 's' : ''}}${{d.is_wicket ? ' (W)' : ''}}`;
            syncStatus.innerText = `● Ball ${{d.over}}.${{d.ball}} (${{d.video_time_sec}}s)`;

            // 3. Append Lead Commentary Card at the Bottom
            const leadPersonaName = d.lead_persona || 'James Sterling (Lead)';
            const leadCard = document.createElement('div');
            leadCard.className = 'card-lead';
            leadCard.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                    <span style="font-size:0.75rem; font-weight:800; color:#00F0FF;">🎙️ ${{leadPersonaName}}</span>
                    <span style="font-size:0.7rem; color:#A0AEC0;">Over ${{d.over}}.${{d.ball}} • ${{d.runs_total}} runs</span>
                </div>
                <div style="font-size:0.84rem; color:#FFFFFF;">${{d.lead_commentary}}</div>
            `;
            commFeed.appendChild(leadCard);

            // 4. Append Analyst Commentary Card (if present)
            if (d.analyst_commentary) {{
                const analystPersonaName = d.analyst_persona || 'Harsha Atherton (Analyst)';
                const analystCard = document.createElement('div');
                analystCard.className = 'card-analyst';
                analystCard.innerHTML = `
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                        <span style="font-size:0.75rem; font-weight:800; color:#FFB800;">📊 ${{analystPersonaName}}</span>
                        <span style="font-size:0.7rem; color:#A0AEC0;">Tactical Insight</span>
                    </div>
                    <div style="font-size:0.84rem; color:#FFFFFF;">${{d.analyst_commentary}}</div>
                `;
                commFeed.appendChild(analystCard);
            }}

            // 5. Smooth Auto-Scroll to the Bottom (Bottom-Stacking)
            commFeed.scrollTop = commFeed.scrollHeight;

            // 6. Spoken Audio Narration Played Out Loud Immediately
            const activeAudioUrl = d.dual_audio_url || d.lead_audio_url;
            if (activeAudioUrl && isLivePlaying) {{
                const audioUrl = activeAudioUrl.startsWith("http") ? activeAudioUrl : baseOrigin + activeAudioUrl;
                audio.src = audioUrl;
                audioIndicator.innerText = "🔊 Speaking...";
                audioIndicator.style.color = "#FFD700";
                
                audio.play().then(() => {{
                    audio.onended = () => {{
                        if (isLivePlaying) {{
                            audioIndicator.innerText = "🔊 Voice Ready";
                            audioIndicator.style.color = "#00E5FF";
                        }}
                    }};
                }}).catch(e => {{
                    console.log('Audio playback notice:', e);
                }});
            }}
        }}

        // Continuous 60 FPS Video Timeline Tracker (8.0-Second Post-Delivery Broadcast Delay)
        video.addEventListener('timeupdate', () => {{
            if (!isLivePlaying) return;
            const curTime = video.currentTime;
            
            for (let i = 0; i < deliveries.length; i++) {{
                const d = deliveries[i];
                const bTime = (d.broadcast_time_sec !== undefined && d.broadcast_time_sec !== null) ? d.broadcast_time_sec : (d.video_time_sec + 8.0);
                if (d.video_time_sec !== null && curTime >= bTime - 0.2 && curTime <= bTime + 3.0) {{
                    triggerDeliveryEvent(d, i);
                }}
            }}
        }});

        // Play / Pause Stream Toggle
        function togglePlayLiveStream() {{
            if (!isLivePlaying) {{
                isLivePlaying = true;
                playBtn.innerText = "⏸ Pause Live Stream";
                playBtn.style.background = "linear-gradient(135deg, #FF0055 0%, #FF5500 100%)";
                playBtn.style.color = "#FFFFFF";
                
                if (hdrBadge) {{
                    hdrBadge.innerText = "● LIVE BROADCAST";
                    hdrBadge.classList.remove("paused");
                }}

                if (video.currentTime < 24.0) {{
                    video.currentTime = 24.0;
                }}
                video.play().catch(e => {{
                    console.log('Video playback error:', e);
                }});

                // Trigger current delivery audio if available
                if (currentDeliveryIdx < deliveries.length) {{
                    const curD = deliveries[Math.max(0, currentDeliveryIdx - 1)];
                    if (curD && curD.lead_audio_url && audio.paused) {{
                        const audioUrl = curD.lead_audio_url.startsWith("http") ? curD.lead_audio_url : baseOrigin + curD.lead_audio_url;
                        audio.src = audioUrl;
                        audio.play().catch(e => console.log('Audio play error:', e));
                    }}
                }}
            }} else {{
                // Pause clean halt: stops both video and audio
                isLivePlaying = false;
                playBtn.innerText = "▶ Play Live Stream";
                playBtn.style.background = "linear-gradient(135deg, #00F0FF 0%, #0088FF 100%)";
                playBtn.style.color = "#070B19";
                
                if (hdrBadge) {{
                    hdrBadge.innerText = "⏸ PAUSED";
                    hdrBadge.classList.add("paused");
                }}

                video.pause();
                audio.pause();
                audioIndicator.innerText = "🔇 Stream Paused";
                audioIndicator.style.color = "#718096";
            }}
        }}

        // Step Next Delivery (Starts video at ball delivery release, updates scorecard + audio at 8.0s delay)
        function stepNextDelivery() {{
            if (currentDeliveryIdx >= deliveries.length) {{
                alert("Match replay completed!");
                return;
            }}
            const nextDeliv = deliveries[currentDeliveryIdx];
            const nextIdx = currentDeliveryIdx;
            if (nextDeliv.video_time_sec !== null) {{
                video.currentTime = nextDeliv.video_time_sec;
                isLivePlaying = true;
                playBtn.innerText = "⏸ Pause Live Stream";
                playBtn.style.background = "linear-gradient(135deg, #FF0055 0%, #FF5500 100%)";
                playBtn.style.color = "#FFFFFF";
                if (hdrBadge) {{
                    hdrBadge.innerText = "● LIVE BROADCAST";
                    hdrBadge.classList.remove("paused");
                }}
                video.play().catch(e => console.log(e));
                setTimeout(() => {{
                    triggerDeliveryEvent(nextDeliv, nextIdx);
                }}, 8000);
            }}
        }}

        // Reset to Over 18.0 (Atomic reset across all components)
        function resetToOver18() {{
            isLivePlaying = false;
            triggeredDeliveries.clear();
            currentDeliveryIdx = 0;
            video.currentTime = 24.0;
            video.pause();
            audio.pause();
            
            playBtn.innerText = "▶ Play Live Stream";
            playBtn.style.background = "linear-gradient(135deg, #00F0FF 0%, #0088FF 100%)";
            playBtn.style.color = "#070B19";
            
            if (hdrBadge) {{
                hdrBadge.innerText = "● LIVE BROADCAST";
                hdrBadge.classList.remove("paused");
            }}

            // Reset Header Scoreboard
            if (hdrScore) hdrScore.innerText = "{initial_score}";
            if (hdrOvers) hdrOvers.innerText = "({initial_overs} ov)";
            if (hdrStriker) hdrStriker.innerText = "{initial_striker}";
            if (hdrBowler) hdrBowler.innerText = "{initial_bowler}";
            if (hdrCrr) hdrCrr.innerText = "CRR: {initial_crr}";

            // Reset TV Overlay
            scoreOverlay.innerText = "ENG {initial_score} ({initial_overs} ov)";
            playersOverlay.innerText = "🏏 {initial_striker} | 🎯 {initial_bowler}";
            lastBallOverlay.innerText = "RESET (24.0s)";
            syncStatus.innerText = "● Over 18 Ready";
            audioIndicator.innerText = "🔊 Voice Ready";
            audioIndicator.style.color = "#00E5FF";

            // Reset Commentary Feed
            commFeed.innerHTML = `
                <div style="background: rgba(0, 240, 255, 0.05); border-left: 3px solid #00F0FF; border-radius: 6px; padding: 8px 10px; font-size: 0.82rem; color: #A0AEC0;">
                    Reset to Over 18.0 (24.0s). Click <strong>Play Live Stream</strong> or <strong>Next Delivery</strong> to begin continuous commentary.
                </div>
            `;
        }}
    </script>
    </body>
    </html>
    """
