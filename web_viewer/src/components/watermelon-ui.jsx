// Watermelon UI registry primitives, adapted from the registry's copy-paste
// Card, Badge, Button, Progress, and Separator components for this Vite app.

function joinClasses(...values) {
  return values.filter(Boolean).join(" ");
}

export function Card({ className, ...props }) {
  return <div data-slot="card" className={joinClasses("wm-card", className)} {...props} />;
}

export function CardHeader({ className, ...props }) {
  return <div data-slot="card-header" className={joinClasses("wm-card-header", className)} {...props} />;
}

export function CardTitle({ className, ...props }) {
  return <div data-slot="card-title" className={joinClasses("wm-card-title", className)} {...props} />;
}

export function CardDescription({ className, ...props }) {
  return <div data-slot="card-description" className={joinClasses("wm-card-description", className)} {...props} />;
}

export function CardContent({ className, ...props }) {
  return <div data-slot="card-content" className={joinClasses("wm-card-content", className)} {...props} />;
}

export function CardFooter({ className, ...props }) {
  return <div data-slot="card-footer" className={joinClasses("wm-card-footer", className)} {...props} />;
}

export function Badge({ variant = "default", className, ...props }) {
  return <span data-slot="badge" className={joinClasses("wm-badge", `wm-badge-${variant}`, className)} {...props} />;
}

export function Button({ variant = "default", className, ...props }) {
  return <button data-slot="button" className={joinClasses("wm-button", `wm-button-${variant}`, className)} {...props} />;
}

export function Progress({ value = 0, className, ...props }) {
  return (
    <div data-slot="progress" className={joinClasses("wm-progress", className)} {...props}>
      <span style={{ width: `${Math.min(Math.max(Number(value) || 0, 0), 100)}%` }} />
    </div>
  );
}

export function Separator({ className, ...props }) {
  return <div data-slot="separator" role="separator" className={joinClasses("wm-separator", className)} {...props} />;
}
