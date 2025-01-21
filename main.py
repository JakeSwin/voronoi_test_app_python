import freud
import pyray as pr
from pyray import ffi
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')

from skimage.draw import polygon2mask
from pykrige.ok import OrdinaryKriging

WINDOW_SIZE = 800

pr.init_window(WINDOW_SIZE, WINDOW_SIZE, "Weighted Voronoi GP")
pr.set_target_fps(30)

camera = pr.Camera2D()
camera.target = pr.Vector2(0, 0)

def find_centroid(p, a):
  p = np.append(p, p[0, np.newaxis], axis=0)
  cx = 0
  cy = 0
  for i in range(len(p) - 1) :
    cx += (p[i][0] + p[i+1][0]) * (p[i][0]*p[i+1][1] - p[i+1][0]*p[i][1])
    cy += (p[i][1] + p[i+1][1]) * (p[i][0]*p[i+1][1] - p[i+1][0]*p[i][1])
  cx *= 1 / (6 * a)
  cy *= 1 / (6 * a)
  return (cx, cy)

def point_to_image_transform(Xs, Ys, img_size):
    new_x = -Xs + (img_size / 2)
    new_y = -(-Ys - (img_size / 2))
    return np.column_stack((new_x, new_y))

def image_to_point_transform(new_x, new_y, img_size):
    Xs = (new_x - (img_size / 2))
    Ys = -new_y + (img_size / 2)
    return np.column_stack((Xs, Ys))

if __name__ == "__main__":
    im = Image.open("/home/swin/code/python/voronoi/000/000/groundtruth/first000_gt.png")
    width, height = im.size
    crop_size = 150
    num_samples = 2000
    x = np.random.randint(low=crop_size, high=width-crop_size+1, size=num_samples)
    y = np.random.randint(low=crop_size, high=height-crop_size+1, size=num_samples)
    weed_chance = np.zeros(num_samples)
    for i in range(num_samples):
        crop = im.crop((x[i], y[i], x[i]+crop_size, y[i]+crop_size))
        weed_chance[i] = np.count_nonzero(np.array(crop)[:, :, 0]) / crop_size**2

    OK = OrdinaryKriging(
        x,
        y,
        weed_chance,
        variogram_model='exponential',
        verbose=True,
        enable_plotting=False,
    )

    gridx = np.arange(0, width, 45, dtype='float64')
    gridy = np.arange(0, width, 45, dtype='float64')
    zstar, ss = OK.execute("grid", gridx, gridy)
    normalized_zstar = (zstar.data - np.min(zstar.data)) / (np.max(zstar.data) - np.min(zstar.data))
    # fig = plt.figure()
    # cax = plt.imshow(normalized_zstar)
    # ax = plt.gca()
    # cbar = plt.colorbar(cax)
    # plt.show()
    # GP size
    gp_size = zstar.data.shape[0]

    # Create sample points
    points = np.random.uniform(-gp_size/2, gp_size/2, size=(200, 3))
    points[:, 2] = 0
    original_points = np.copy(points)

    # Create a box and Voronoi compute object
    L = gp_size # was 2
    box = freud.box.Box.square(L)
    voro = freud.locality.Voronoi()

    # Create resized image for display with raylib
    resized_im = im.resize((WINDOW_SIZE, WINDOW_SIZE))
    # colors = [pr.Color for _ in range(WINDOW_SIZE * WINDOW_SIZE)]
    colors = ffi.new("Color[]", WINDOW_SIZE * WINDOW_SIZE)
    
    # Convert pixel data
    pixels = resized_im.load()
    for y in range(WINDOW_SIZE):
        for x in range(WINDOW_SIZE):
            r, g, b = pixels[x, y]
            colors[y * WINDOW_SIZE + x] = pr.Color(r, g, b, 255)
    
    # Create raylib Image
    raylib_image = pr.Image(
        colors,
        WINDOW_SIZE,
        WINDOW_SIZE,
        1,
        pr.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
    )

    texture = pr.load_texture_from_image(raylib_image)
    # pr.unload_image(raylib_image)

    point_increase_ratio = WINDOW_SIZE / gp_size 

    while not pr.window_should_close():
        # Compute the Voronoi diagram
        voro.compute((box, points))

        # Compute the Centroids of Voronois
        centroids = np.array([find_centroid(points, area) for points, area in zip(voro.polytopes, voro.volumes)])
        w_centroids = np.zeros(shape=centroids.shape)

        # Compute the masks of Voronois
        masks = [polygon2mask((gp_size, gp_size), point_to_image_transform(polytope[:, 1], polytope[:, 0], gp_size)) for polytope in voro.polytopes]

        # # Shift centroids towards high interest regions
        for j in range(len(centroids)):
            centroid = np.copy(centroids[j])

            # Get indices and values
            y_idxs, x_idxs = np.where(masks[j])
            masked_values = np.expand_dims(normalized_zstar[masks[j]], axis=1)
            masked_values[masked_values < 0] = 0
            # masked_values = masked_values**-1
            masked_values = 1 - masked_values
            # if masked_values.size > 0:
                # masked_values = (masked_values - np.min(masked_values)) / (np.max(masked_values) - np.min(masked_values))

            # Combine into array of (y, x, value) tuples
            # result = np.column_stack((y_idxs, x_idxs, masked_values))
            # result = np.append(image_to_point_transform(x_idxs, y_idxs, gp_size), masked_values, axis=1)
            # total_weight = np.sum(result[:, 2])
            coords = image_to_point_transform(x_idxs, y_idxs, gp_size)
            # masked_values = np.column_stack((
            #     np.where(coords[:, 0] < centroid[0], -masked_values.flatten(), masked_values.flatten()),
            #     np.where(coords[:, 1] < centroid[1], -masked_values.flatten(), masked_values.flatten())
            # ))
            result = np.append(coords, masked_values, axis=1)

            total_weight = np.sum(masked_values)

            # for point in result:
            #   centroid[0] += (point[0] * point[2])
            #   centroid[1] += (point[1] * point[2])

            if total_weight > 0:
            # centroid /= total_weight
            # w_centroids[i] = centroid
                w_centroids[j] = (centroid + np.sum(coords * masked_values, axis=0)) / total_weight
            else:
                w_centroids[j] = points[j][0:2]

            # print(f"Iter: i"w_centroids[i])

        pr.begin_drawing()
        pr.clear_background(pr.RAYWHITE)
        pr.draw_texture(texture, 0, 0, pr.WHITE)

        for polytope in voro.polytopes:
            polytope_points = point_to_image_transform(polytope[:, 0], polytope[:, 1], gp_size) * point_increase_ratio
            polytope_points = np.append(polytope_points, np.expand_dims(polytope_points[0], axis=0), axis=0).astype(np.int32)
            for pi in range(len(polytope_points) - 1):
                pr.draw_line(int(polytope_points[pi][0]), int(polytope_points[pi][1]), polytope_points[pi+1][0], polytope_points[pi+1][1], pr.WHITE)

        for point in (point_to_image_transform(points[:, 0], points[:, 1], gp_size) * point_increase_ratio).astype(np.int32):
            pr.draw_circle(point[0], point[1], 2, pr.WHITE)

        for point in (point_to_image_transform(w_centroids[:, 0], w_centroids[:, 1], gp_size) * point_increase_ratio).astype(np.int32):
            pr.draw_circle(point[0], point[1], 2, pr.BLUE)

        pr.end_drawing()

        # Compute new points through linear interpolation
        points = np.array([p1 + 0.1 * (p2 - p1) for p1, p2 in zip(points[:, 0:2], w_centroids)])
        points = np.append(points, np.zeros((len(points),1)), axis=1)
    
    pr.close_window()

    # # Plot the results
    # plt.figure(figsize=(10, 10))
    # ax = plt.gca()
    # voro.plot(ax=ax, cmap="RdBu")
    # ax.scatter(points[:, 0], points[:, 1], s=2, c="k")
    # ax.scatter(original_points[:, 0], original_points[:, 1], s=2, c="g")
    # ax.scatter(centroids[:, 0], centroids[:, 1], s=2, c="r")
    # ax.scatter(w_centroids[:, 0], w_centroids[:, 1], s=2, c="b")
    # cax = ax.imshow(normalized_zstar, extent=(-gp_size/2, gp_size/2, -gp_size/2, gp_size/2))
    # cbar = plt.colorbar(cax)
    # plt.show()
