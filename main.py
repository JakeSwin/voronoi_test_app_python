import freud
import pyray as pr
from pyray import ffi
import numpy as np
import PIL.Image as Image

from skimage.draw import polygon2mask
from pykrige.ok import OrdinaryKriging

WINDOW_SIZE = 800
COLOUR_1 = [75, 0, 130]
COLOUR_2 = [255, 255, 0]
SHOW_LINES = True
SHOW_COLOURS = False
SHOW_POINTS = True

pr.init_window(WINDOW_SIZE, WINDOW_SIZE, "Weighted Voronoi GP")
pr.set_target_fps(30)

camera = pr.Camera2D()
camera.target = pr.Vector2(0, 0)

def interpolate(color1, color2, fraction):
    return [int(c1 + (c2 - c1) * fraction) for c1, c2 in zip(color1, color2)]

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

def generate_points(num_points, prob_map, gp_size):
    rng = np.random.default_rng()
    Xs = []
    Ys = []
    while len(Xs) < num_points:
        x_coord = rng.uniform(-gp_size/2, gp_size/2)
        y_coord = rng.uniform(-gp_size/2, gp_size/2)
        sample_coords = point_to_image_transform(y_coord, x_coord, gp_size)
        sampled_prob = prob_map[int(sample_coords[0][0])][int(sample_coords[0][1])]
        if rng.uniform() < sampled_prob:
            Xs.append(x_coord)
            Ys.append(y_coord)
    return np.column_stack((Xs, Ys))

if __name__ == "__main__":
    im = Image.open("/home/swin/code/python/voronoi/000/groundtruth/first000_gt.png")
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

    gridx = np.arange(0, width, 40, dtype='float64')
    gridy = np.arange(0, width, 40, dtype='float64')
    zstar, ss = OK.execute("grid", gridx, gridy)
    normalized_zstar = (zstar.data - np.min(zstar.data)) / (np.max(zstar.data) - np.min(zstar.data))
    # GP size
    gp_size = zstar.data.shape[0]

    # Create sample points
    # points = np.random.uniform(-gp_size/2, gp_size/2, size=(250, 3))
    points = generate_points(400, normalized_zstar, gp_size)
    points = np.append(points, np.expand_dims(np.zeros(len(points)), axis=1), axis=1)
    points[:, 2] = 0
    original_points = np.copy(points)

    # Create a box and Voronoi compute object
    L = gp_size # was 2
    box = freud.box.Box.square(L)
    voro = freud.locality.Voronoi()

    # Create resized image for display with raylib
    resized_im = im.resize((WINDOW_SIZE, WINDOW_SIZE))
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

    point_increase_ratio = WINDOW_SIZE / gp_size

    while not pr.window_should_close():
        if pr.is_key_pressed(pr.KEY_R):
            points = generate_points(400, normalized_zstar, gp_size)
            points = np.append(points, np.expand_dims(np.zeros(len(points)), axis=1), axis=1)
            points[:, 2] = 0
            original_points = np.copy(points)

        if pr.is_key_pressed(pr.KEY_P):
            SHOW_POINTS = not SHOW_POINTS

        if pr.is_key_pressed(pr.KEY_L):
            SHOW_LINES = not SHOW_LINES

        if pr.is_key_pressed(pr.KEY_C):
            SHOW_COLOURS = not SHOW_COLOURS

        # Compute the Voronoi diagram
        voro.compute((box, points))

        # Compute the Centroids of Voronois
        centroids = np.array([find_centroid(points, area) for points, area in zip(voro.polytopes, voro.volumes)])
        w_centroids = np.zeros(shape=centroids.shape)

        # Compute the masks of Voronois
        masks = [polygon2mask((gp_size, gp_size), point_to_image_transform(polytope[:, 1], polytope[:, 0], gp_size)) for polytope in voro.polytopes]

        # Create array to hold average weight of region
        avg_weights = np.zeros(len(points))

        # # Shift centroids towards high interest regions
        for j in range(len(centroids)):
            centroid = np.copy(centroids[j])

            # Get indices and values
            y_idxs, x_idxs = np.where(masks[j])
            masked_values = np.expand_dims(normalized_zstar[masks[j]], axis=1)
            masked_values[masked_values < 0] = 0
            coords = image_to_point_transform(x_idxs, y_idxs, gp_size)
            result = np.append(coords, masked_values, axis=1)

            total_weight = np.sum(masked_values)
            if total_weight > 0 and len(masked_values) > 1:
                avg_weight = total_weight / len(masked_values)
                avg_weights[j] = avg_weight
            else:
                avg_weights[j] = 0.0

            if total_weight > 0:
                w_centroids[j] = np.sum(coords * masked_values, axis=0) / total_weight
            else:
                w_centroids[j] = points[j][0:2]

        pr.begin_drawing()
        pr.clear_background(pr.RAYWHITE)
        pr.draw_texture(texture, 0, 0, pr.WHITE)

        if SHOW_COLOURS:
            for i, c in enumerate(centroids):
                triangle_points = np.insert(voro.polytopes[i][:, 0:2], 0, c, axis=0)
                triangle_points = np.append(triangle_points, np.expand_dims(triangle_points[1], axis=0), axis=0)
                triangle_points = point_to_image_transform(triangle_points[:, 0], triangle_points[:, 1], gp_size) * point_increase_ratio
                triangle_points = triangle_points.astype(np.int32).tolist()
                region_color = interpolate(COLOUR_1, COLOUR_2, avg_weights[i])
                pr.draw_triangle_fan(triangle_points, len(triangle_points), pr.Color(region_color[0], region_color[1], region_color[2], 255))

        if SHOW_LINES:
            for polytope in voro.polytopes:
                polytope_points = point_to_image_transform(polytope[:, 0], polytope[:, 1], gp_size) * point_increase_ratio
                polytope_points = np.append(polytope_points, np.expand_dims(polytope_points[0], axis=0), axis=0).astype(np.int32)
                for pi in range(len(polytope_points) - 1):
                    pr.draw_line(int(polytope_points[pi][0]), int(polytope_points[pi][1]), polytope_points[pi+1][0], polytope_points[pi+1][1], pr.WHITE)

        if SHOW_POINTS:
            for point in (point_to_image_transform(points[:, 0], points[:, 1], gp_size) * point_increase_ratio).astype(np.int32):
                pr.draw_circle(point[0], point[1], 2, pr.WHITE)

            for point in (point_to_image_transform(w_centroids[:, 0], w_centroids[:, 1], gp_size) * point_increase_ratio).astype(np.int32):
                pr.draw_circle(point[0], point[1], 2, pr.BLUE)

        pr.end_drawing()

        # Compute new points through linear interpolation
        points = np.array([p1 + 0.1 * (p2 - p1) for p1, p2 in zip(points[:, 0:2], w_centroids)])
        points = np.append(points, np.zeros((len(points),1)), axis=1)

    pr.close_window()
