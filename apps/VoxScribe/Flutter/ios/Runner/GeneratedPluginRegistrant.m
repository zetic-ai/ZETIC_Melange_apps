//
//  Generated file. Do not edit.
//

// clang-format off

#import "GeneratedPluginRegistrant.h"

#if __has_include(<audioplayers_darwin/AudioplayersDarwinPlugin.h>)
#import <audioplayers_darwin/AudioplayersDarwinPlugin.h>
#else
@import audioplayers_darwin;
#endif

#if __has_include(<zetic_mlange/MlangeFlutterPlugin.h>)
#import <zetic_mlange/MlangeFlutterPlugin.h>
#else
@import zetic_mlange;
#endif

@implementation GeneratedPluginRegistrant

+ (void)registerWithRegistry:(NSObject<FlutterPluginRegistry>*)registry {
  [AudioplayersDarwinPlugin registerWithRegistrar:[registry registrarForPlugin:@"AudioplayersDarwinPlugin"]];
  [MlangeFlutterPlugin registerWithRegistrar:[registry registrarForPlugin:@"MlangeFlutterPlugin"]];
}

@end
