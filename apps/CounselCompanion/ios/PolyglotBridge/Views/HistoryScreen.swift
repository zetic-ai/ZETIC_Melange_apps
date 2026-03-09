import SwiftUI

struct HistoryScreen: View {
    @EnvironmentObject private var app: AppViewModel
    let onSelect: (Session) -> Void

    var body: some View {
        NavigationView {
            List {
                ForEach(app.sessions) { session in
                    Button(action: { onSelect(session) }) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(session.title.isEmpty ? "Session" : session.title)
                                .font(.headline)
                            Text(session.messages.last?.content.prefix(80) ?? "")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    .swipeActions {
                        Button(role: .destructive) {
                            app.deleteSession(session.id)
                        } label: { Label("Delete", systemImage: "trash") }
                    }
                }
            }
            .navigationTitle("History")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: app.createNewSession) { Label("New", systemImage: "plus") }
                }
            }
        }
    }
}
