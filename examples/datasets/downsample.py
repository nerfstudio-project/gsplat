import os


if __name__ == "__main__":
    import argparse

    import mediapy as media
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    dir_suffix = f"_{args.factor}"
    input_image_dir = os.path.join(args.data_dir, "images")
    image_dir = os.path.join(args.data_dir, "images" + dir_suffix)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        im_files = os.listdir(input_image_dir)
        for im_file in im_files:
            im = media.read_image(os.path.join(input_image_dir, im_file))

            resized_im = media.resize_image(
                im, (im.shape[0] // args.factor, im.shape[1] // args.factor)
            )
            media.write_image(os.path.join(image_dir, im_file), resized_im)
    else:
        print(f"{image_dir} already exists! no downsampling.")
