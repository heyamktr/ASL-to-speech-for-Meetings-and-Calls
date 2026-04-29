export default function Popup() {
  return (
    <div className="space-y-3">
      <h1 className="text-base font-semibold">ASL-to-Speech</h1>
      <label className="flex items-center justify-between">
        <span className="text-sm">Enabled</span>
        <input type="checkbox" defaultChecked />
      </label>
      <p className="text-xs text-gray-500">Settings stub — wire up storage next.</p>
    </div>
  );
}
