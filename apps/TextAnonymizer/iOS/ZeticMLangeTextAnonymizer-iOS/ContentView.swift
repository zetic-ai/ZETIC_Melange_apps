//
//  ContentView.swift
//  TextAnonymizer
//
//  Main UI for text anonymization
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = AnonymizerViewModel()
    @State private var inputText = ""
    @State private var showingShareSheet = false
    @FocusState private var inputFocused: Bool
    
    // Zetic Brand Color #34A9A3
    private let zeticTeal = Color(red: 52/255, green: 169/255, blue: 163/255)
    
    var body: some View {
        ZStack {
            // Background Gradient
            LinearGradient(
                colors: [zeticTeal.opacity(0.9), zeticTeal.opacity(0.7)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            ScrollView {
                VStack(spacing: 20) {
                    headerView
                    mainCardView
                    Spacer(minLength: 24)
                }
            }
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        inputFocused = false
                    }
                }
            }
            .sheet(isPresented: $showingShareSheet) {
                ShareSheet(items: [viewModel.anonymizedText])
            }
            .alert("Error", isPresented: $viewModel.showingError) {
                Button("OK") {
                    viewModel.showingError = false
                }
            } message: {
                Text(viewModel.errorMessage)
            }
        }
    }
    
    private var headerView: some View {
        VStack(spacing: 4) {
            Text("tanaos-text-anonymizer-v1")
                .font(.title2.bold())
                .foregroundColor(.white)
            
            Text("Powered by MLange")
                .font(.subheadline)
                .foregroundColor(.white.opacity(0.9))
        }
        .padding(.top, 20)
        .padding(.bottom, 10)
    }
    
    private var mainCardView: some View {
        VStack(alignment: .leading, spacing: 16) {
            modelLoadingView
            exampleButtonsView
            inputSectionView
            anonymizeButton
            outputSectionView
        }
        .padding(20)
        .background(Color.white)
        .cornerRadius(20)
        .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 5)
        .padding(.horizontal, 16)
    }
    
    private var modelLoadingView: some View {
        Group {
            if !viewModel.isModelLoaded {
                HStack {
                    Spacer()
                    VStack {
                        HStack {
                            ProgressView()
                                .tint(.gray)
                            Text(viewModel.downloadProgress > 0 ? "Downloading... \(viewModel.downloadProgress)%" : "Loading model...")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                    Spacer()
                }
                .padding(.bottom, 4)
            }
        }
    }
    
    private var exampleButtonsView: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(1...10, id: \.self) { i in
                    Button("Example \(i)") {
                        inputText = getExampleText(for: i)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray6))
                    .cornerRadius(16)
                    .foregroundColor(.primary)
                }
            }
        }
    }
    
    private var inputSectionView: some View {
        Group {
            Text("Paste your text here:")
                .font(.headline)
                .foregroundColor(.secondary)
            
            TextEditor(text: $inputText)
                .frame(height: 150)
                .padding(8)
                .background(Color(.systemGray6))
                .cornerRadius(12)
                .focused($inputFocused)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
        }
    }
    
    private var anonymizeButton: some View {
        Button(action: {
            viewModel.anonymizeText(inputText)
        }) {
            HStack {
                if viewModel.isProcessing {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                }
                Text(viewModel.isProcessing ? "Anonymizing..." : "Anonymize Text")
                    .font(.headline)
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(inputText.isEmpty || viewModel.isProcessing || !viewModel.isModelLoaded ? Color.gray : zeticTeal)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(inputText.isEmpty || viewModel.isProcessing || !viewModel.isModelLoaded)
    }
    
    private var outputSectionView: some View {
        Group {
            if !viewModel.anonymizedText.isEmpty {
                Divider()
                    .padding(.vertical, 8)
                
                Text("Anonymized Text:")
                    .font(.headline)
                    .foregroundColor(.secondary)
                
                ScrollView {
                    Text(viewModel.anonymizedText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                }
                .frame(height: 150)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
                
                // Copy and Share Buttons
                HStack(spacing: 12) {
                    Button(action: {
                        UIPasteboard.general.string = viewModel.anonymizedText
                    }) {
                        HStack {
                            Image(systemName: "doc.on.doc")
                            Text("Copy")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    
                    Button(action: {
                        showingShareSheet = true
                    }) {
                        HStack {
                            Image(systemName: "square.and.arrow.up")
                            Text("Share")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                }
            }
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: items,
            applicationActivities: nil
        )
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

private func getExampleText(for index: Int) -> String {
    switch index {
    case 1: return "Mr. Sherlock Holmes received a visitor at 221B Baker Street, London on October 31st, 1890. The client, Jabez Wilson, had a strange story about a Red-Headed League which dissolved unexpectedly."
    case 2: return "Tony Stark held a press conference in New York on May 2, 2008. He declared 'I am Iron Man' to the reporters, causing Stark Industries stock to rise by 15% immediately after the announcement."
    case 3: return "Harry Potter received his first letter at 4 Privet Drive, Little Whinging, Surrey on his 11th birthday, July 31, 1991. Rubeus Hagrid later arrived to personally deliver the invitation to Hogwarts."
    case 4: return "Jack Dawson won his ticket to the Titanic in a highly lucky poker game on April 10, 1912. The ship departed from Southampton shortly after and was scheduled to arrive in New York City."
    case 5: return "Project Manager Sarah Connor called 555-0199 to schedule an urgent meeting with Miles Dyson. They agreed to meet at 123 Cyberdyne Systems Way, California on August 29th to discuss the neural net processor."
    case 6: return "Patient John Doe (ID: 998-877-66) was admitted to Princeton-Plainsboro Teaching Hospital on November 21st. Dr. Gregory House ordered an MRI and a lumbar puncture immediately despite the team's initial hesitation."
    case 7: return "On June 17, 1994, a white Ford Bronco was driven by Al Cowlings in Los Angeles. The passenger, O.J. Simpson, was subsequently charged, and the trial began on January 24, 1995."
    case 8: return "Please transfer 1,000,000 USD to the account 1234-5678-9012 held by Walter White. The transaction must be completed by December 25th at the Albuquerque branch to avoid suspicion from the DEA."
    case 9: return "Order ID 556677 for Frodo Baggins cannot be delivered to Bag End, Hobbiton due to the recipient's absence. Please reroute the package to the Prancing Pony Inn, Bree, by September 22nd."
    case 10: return "On July 20, 1969, Neil Armstrong and Buzz Aldrin landed the Eagle on the Moon surface. They planted the US flag and returned to Earth, splashing down in the Pacific Ocean on July 24th."
    default: return ""
    }
}
