// One event, fully told: the evidence clip plus every measured number.

import { api, type NearMissEvent } from "../api";

export function EventDetail({ event }: { event: NearMissEvent | null }) {
  if (!event)
    return <div className="empty">Select an event to replay it.</div>;
  return (
    <div className="detail">
      {event.has_clip ? (
        <video key={event.id} src={api.clipUrl(event.id)} controls autoPlay muted loop />
      ) : (
        <div className="empty">No clip stored for this event.</div>
      )}
      <div className="metrics">
        <div>
          <div className="k">participants</div>
          <div className="v">
            {event.class_a} × {event.class_b}
          </div>
        </div>
        <div>
          <div className="k">risk score</div>
          <div className="v">{event.risk_score.toFixed(0)} / 100</div>
        </div>
        <div>
          <div className="k">min time to collision</div>
          <div className="v">{event.min_ttc_s != null ? `${event.min_ttc_s}s` : "—"}</div>
        </div>
        <div>
          <div className="k">post encroachment</div>
          <div className="v">{event.pet_s != null ? `${event.pet_s}s` : "—"}</div>
        </div>
        <div>
          <div className="k">closest distance</div>
          <div className="v">{event.min_distance_m}m</div>
        </div>
        <div>
          <div className="k">max closing speed</div>
          <div className="v">{event.max_closing_kmh} km/h</div>
        </div>
      </div>
    </div>
  );
}
