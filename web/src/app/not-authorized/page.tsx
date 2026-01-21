export default function NotAuthorized() {
  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <div className="max-w-md rounded-2xl border p-6 shadow-sm">
        <h1 className="text-xl font-semibold">Not authorized</h1>
        <p className="mt-2 text-sm text-gray-600">
          Your account is not on the reviewer allowlist. Ask the admin for access.
        </p>
      </div>
    </main>
  );
}
