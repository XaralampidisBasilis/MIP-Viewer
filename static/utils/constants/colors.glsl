#ifndef COLOR_CONSTANTS
#define COLOR_CONSTANTS

struct ColorConstants {
    // Basic
    vec3 BLACK;
    vec3 WHITE;
    vec3 GRAY;
    vec3 LIGHT_GRAY;
    vec3 DARK_GRAY;
    vec3 TRANSPARENT;

    // Primary
    vec3 RED;
    vec3 GREEN;
    vec3 BLUE;

    vec3 LIGHT_RED;
    vec3 LIGHT_GREEN;
    vec3 LIGHT_BLUE;
    
    vec3 DARK_RED;
    vec3 DARK_GREEN;
    vec3 DARK_BLUE;

    // Secondary
    vec3 CYAN;
    vec3 MAGENTA;
    vec3 YELLOW;
    vec3 LIGHT_CYAN;
    vec3 LIGHT_MAGENTA;
    vec3 LIGHT_YELLOW;
    vec3 DARK_CYAN;
    vec3 DARK_MAGENTA;
    vec3 DARK_YELLOW;

    // Pastels
    vec3 PASTEL_RED;
    vec3 PASTEL_GREEN;
    vec3 PASTEL_BLUE;
    vec3 PASTEL_CYAN;
    vec3 PASTEL_MAGENTA;
    vec3 PASTEL_YELLOW;
    vec3 PASTEL_ORANGE;
    vec3 PASTEL_PINK;
    vec3 PASTEL_PURPLE;

    // Extended
    vec3 ORANGE;
    vec3 PINK;
    vec3 PURPLE;
    vec3 BROWN;
    vec3 TEAL;
    vec3 INDIGO;

    vec3 LIGHT_ORANGE;
    vec3 LIGHT_PINK;
    vec3 LIGHT_PURPLE;
    vec3 LIGHT_BROWN;
    vec3 LIGHT_TEAL;
    vec3 LIGHT_INDIGO;

    vec3 DARK_ORANGE;
    vec3 DARK_PINK;
    vec3 DARK_PURPLE;
    vec3 DARK_BROWN;
    vec3 DARK_TEAL;
    vec3 DARK_INDIGO;

    // Rich tones for geometric shading
    vec3 NAVY;
    vec3 LIME;
    vec3 SAND;
    vec3 SKY;
    vec3 MAROON;
    vec3 FOREST;

    // Metallics
    vec3 GOLD;
    vec3 SILVER;

    // Distinct
    vec3 BREWER_SET1[9];
};

const ColorConstants COLOR = ColorConstants(
    // Basic
    vec3(0.0, 0.0, 0.0),       // BLACK
    vec3(1.0, 1.0, 1.0),       // WHITE
    vec3(0.5, 0.5, 0.5),       // GRAY
    vec3(0.8, 0.8, 0.8),       // LIGHT_GRAY
    vec3(0.3, 0.3, 0.3),       // DARK_GRAY
    vec3(0.0, 0.0, 0.0),       // TRANSPARENT

    // Primary
    vec3(1.0, 0.0, 0.0),       // RED
    vec3(0.0, 1.0, 0.0),       // GREEN
    vec3(0.0, 0.0, 1.0),       // BLUE

    vec3(1.0, 0.4, 0.4),       // LIGHT_RED
    vec3(0.5, 1.0, 0.5),       // LIGHT_GREEN
    vec3(0.4, 0.6, 1.0),       // LIGHT_BLUE

    vec3(0.4, 0.05, 0.05),     // DARK_RED
    vec3(0.0, 0.3, 0.0),       // DARK_GREEN
    vec3(0.05, 0.05, 0.4),     // DARK_BLUE

    // Secondary
    vec3(0.0, 1.0, 1.0),       // CYAN
    vec3(1.0, 0.0, 1.0),       // MAGENTA
    vec3(1.0, 1.0, 0.0),       // YELLOW

    vec3(0.6, 1.0, 1.0),       // LIGHT_CYAN
    vec3(1.0, 0.6, 1.0),       // LIGHT_MAGENTA
    vec3(1.0, 1.0, 0.6),       // LIGHT_YELLOW

    vec3(0.0, 0.4, 0.4),       // DARK_CYAN
    vec3(0.4, 0.0, 0.4),       // DARK_MAGENTA
    vec3(0.5, 0.5, 0.0),       // DARK_YELLOW

    // Pastels
    vec3(1.0, 0.6, 0.6),       // PASTEL_RED
    vec3(0.6, 1.0, 0.6),       // PASTEL_GREEN
    vec3(0.6, 0.6, 1.0),       // PASTEL_BLUE
    vec3(0.6, 1.0, 1.0),       // PASTEL_CYAN
    vec3(1.0, 0.6, 1.0),       // PASTEL_MAGENTA
    vec3(1.0, 1.0, 0.6),       // PASTEL_YELLOW
    vec3(1.0, 0.8, 0.6),       // PASTEL_ORANGE
    vec3(1.0, 0.8, 0.9),       // PASTEL_PINK
    vec3(0.8, 0.6, 1.0),       // PASTEL_PURPLE

    // Extended
    vec3(1.0, 0.5, 0.0),    // ORANGE
    vec3(1.0, 0.75, 0.8),   // PINK
    vec3(0.5, 0.0, 0.5),    // PURPLE
    vec3(0.6, 0.3, 0.0),    // BROWN
    vec3(0.0, 0.5, 0.5),    // TEAL
    vec3(0.3, 0.0, 0.5),    // INDIGO

    vec3(1.0, 0.7, 0.3),    // LIGHT_ORANGE
    vec3(1.0, 0.85, 0.9),   // LIGHT_PINK
    vec3(0.8, 0.6, 1.0),    // LIGHT_PURPLE
    vec3(0.8, 0.6, 0.4),    // LIGHT_BROWN
    vec3(0.4, 0.8, 0.8),    // LIGHT_TEAL
    vec3(0.6, 0.4, 0.9),    // LIGHT_INDIGO

    vec3(0.7, 0.3, 0.0),    // DARK_ORANGE
    vec3(0.8, 0.4, 0.5),    // DARK_PINK
    vec3(0.3, 0.0, 0.3),    // DARK_PURPLE
    vec3(0.4, 0.2, 0.0),    // DARK_BROWN
    vec3(0.0, 0.3, 0.3),    // DARK_TEAL
    vec3(0.2, 0.0, 0.4),     // DARK_INDIGO

    // Rich tones
    vec3(0.0, 0.0, 0.3),       // NAVY
    vec3(0.6, 1.0, 0.4),       // LIME
    vec3(0.94, 0.87, 0.72),    // SAND
    vec3(0.6, 0.8, 1.0),       // SKY
    vec3(0.5, 0.0, 0.0),       // MAROON
    vec3(0.13, 0.55, 0.13),    // FOREST

    // Metallics
    vec3(1.0, 0.84, 0.0),      // GOLD
    vec3(0.75, 0.75, 0.75),     // SILVER

    // Color brewer set1
    vec3[9](
    vec3(228.0/255.0,  26.0/255.0,  28.0/255.0),  // #e41a1c
    vec3( 55.0/255.0, 126.0/255.0, 184.0/255.0),  // #377eb8
    vec3( 77.0/255.0, 175.0/255.0,  74.0/255.0),  // #4daf4a
    vec3(152.0/255.0,  78.0/255.0, 163.0/255.0),  // #984ea3
    vec3(255.0/255.0, 127.0/255.0,   0.0/255.0),  // #ff7f00
    vec3(255.0/255.0, 255.0/255.0,  51.0/255.0),  // #ffff33
    vec3(166.0/255.0,  86.0/255.0,  40.0/255.0),  // #a65628
    vec3(247.0/255.0, 129.0/255.0, 191.0/255.0),  // #f781bf
    vec3(153.0/255.0, 153.0/255.0, 153.0/255.0)   // #999999
    )
);

#endif // COLOR_CONSTANTS
