//
//  ContentView.swift
//  PromptGuard
//
//  Two tabs (Classify, History); Settings/Diagnostics in menu; floating frosted tab bar; full-screen layout.
//

import SwiftUI

struct FloatingTabBar: View {
    @Binding var selected: Int

    var body: some View {
        HStack(spacing: 0) {
            tabItem(icon: "shield.checkered", label: "Classify", tag: 0)
                .frame(maxWidth: .infinity)
            tabItem(icon: "clock.arrow.circlepath", label: "History", tag: 1)
                .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
        .shadow(color: .black.opacity(0.2), radius: 16, x: 0, y: 8)
    }

    @ViewBuilder
    private func tabItem(icon: String, label: String, tag: Int) -> some View {
        Button {
            withAnimation(.snappy) { selected = tag }
        } label: {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .symbolVariant(selected == tag ? .fill : .none)
                    .font(.system(size: 20, weight: .medium))
                Text(label)
                    .font(.caption.weight(selected == tag ? .semibold : .regular))
            }
            .foregroundStyle(selected == tag ? AppTheme.accent : Color.secondary)
        }
        .buttonStyle(.plain)
    }
}

struct ContentView: View {
    @State private var selectedTab = 0
    @State private var showSettings = false
    @State private var showDiagnostics = false
    @AppStorage("useDarkTheme") private var useDarkTheme = false

    var body: some View {
        ZStack {
            // 1) Gradient fills entire window and draws under status bar + home indicator (full screen)
            AppTheme.gradientBackground
                .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
                .ignoresSafeArea(.all)

            // 2) Main content (headers with menu are inside each tab view)
            VStack(spacing: 0) {
                Group {
                    if selectedTab == 0 {
                        LiveView(showSettings: $showSettings, showDiagnostics: $showDiagnostics)
                    } else {
                        HistoryView(showSettings: $showSettings, showDiagnostics: $showDiagnostics)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            // 3) Tab bar overlay; bottom padding clears home indicator (e.g. iPhone 15 Pro)
            GeometryReader { geo in
                VStack {
                    Spacer(minLength: 0)
                    FloatingTabBar(selected: $selectedTab)
                        .padding(.horizontal, 40)
                        .padding(.top, 12)
                        .padding(.bottom, geo.safeAreaInsets.bottom + 12)
                }
            }
        }
        .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
        .ignoresSafeArea(.all)
        .sheet(isPresented: $showSettings) {
            NavigationStack {
                SettingsView()
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Done") { showSettings = false }
                        }
                    }
            }
            .preferredColorScheme(useDarkTheme ? .dark : .light)
        }
        .sheet(isPresented: $showDiagnostics) {
            NavigationStack {
                DiagnosticsView()
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Done") { showDiagnostics = false }
                        }
                    }
            }
            .preferredColorScheme(useDarkTheme ? .dark : .light)
        }
        .preferredColorScheme(useDarkTheme ? .dark : .light)
    }
}

#Preview {
    ContentView()
}
