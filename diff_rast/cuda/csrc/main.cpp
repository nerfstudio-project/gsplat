#include "tgaimage.h"
int main() {
    TGAColor red;
    red[2] = 255;
    red[3] = 255;
    TGAImage image(100, 100, TGAImage::RGB);
    image.set(52, 41, red);
    image.flip_vertically(
    ); // i want to have the origin at the left bottom corner of the image
    image.write_tga_file("output.tga");
    return 0;
}
