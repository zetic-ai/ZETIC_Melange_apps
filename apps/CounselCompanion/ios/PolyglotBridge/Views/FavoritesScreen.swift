import SwiftUI

struct FavoritesScreen: View {
    @EnvironmentObject private var app: AppViewModel
    let onUse: (FavoritePhrase) -> Void

    var body: some View {
        NavigationView {
            List {
                if app.favorites.isEmpty {
                    Text("No favorites yet. Star a translation to save it here.")
                        .foregroundColor(.secondary)
                }
                ForEach(app.favorites) { fav in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(fav.text)
                            .font(AppFont.body())
                        HStack {
                            Text("\(fav.source.flag) -> \(fav.target.flag)")
                            Spacer()
                            Button("Use") { onUse(fav) }
                                .buttonStyle(.bordered)
                        }
                        .font(.caption)
                        .foregroundColor(.secondary)
                    }
                    .swipeActions {
                        Button(role: .destructive) { app.removeFavorite(fav.id) } label: { Label("Delete", systemImage: "trash") }
                    }
                }
            }
            .navigationTitle("Favorites")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Clear") { app.clearFavorites() }
                        .disabled(app.favorites.isEmpty)
                }
            }
        }
    }
}
