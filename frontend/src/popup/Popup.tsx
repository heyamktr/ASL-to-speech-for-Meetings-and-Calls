import { useEffect, useState } from "react";
import { getSettings, setSettings } from "../lib/storage";
import { sendToActiveTab } from "../lib/messaging";

export default function Popup() {
  const [enabled, setEnabled] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSettings().then(({ enabled }) => {
      setEnabled(enabled);
      setLoading(false);
    });
  }, []);

  async function toggle() {
    const next = !enabled;
    setEnabled(next);
    await setSettings({ enabled: next });
    sendToActiveTab({ type: "TOGGLE", enabled: next });
  }

  if (loading) return <div className="p-4 text-sm text-gray-500">Loading…</div>;

  return (
    <div className="w-52 p-4 font-sans">
      <h1 className="text-sm font-semibold text-gray-800 mb-3">ASL Interpreter</h1>

      <button
        onClick={toggle}
        className={`w-full py-2 px-4 rounded text-sm font-medium transition-colors ${
          enabled
            ? "bg-red-100 text-red-700 hover:bg-red-200"
            : "bg-green-100 text-green-700 hover:bg-green-200"
        }`}
      >
        {enabled ? "Disable" : "Enable"}
      </button>

      <p className="mt-2 text-xs text-gray-400">
        Status:{" "}
        <span className={enabled ? "text-green-600" : "text-gray-400"}>
          {enabled ? "Active" : "Off"}
        </span>
      </p>

      <p className="mt-3 text-xs text-gray-400">Open Google Meet and enable to begin capturing.</p>
    </div>
  );
}
