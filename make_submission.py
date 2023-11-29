from utils.submission import masks_to_submission

submission_filename = './predictions/unet_grid/netspresso_10_submission.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = './predictions/unet_grid/pred_test_' + str(i) + '.png'
    print(image_filename)
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)

# t_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# for t in t_list:
#     submission_filename = f'./predictions/unet_grid/netspresso_submission_t{t}.csv'
#     image_filenames = []
#     for i in range(1, 51):
#         image_filename = './predictions/unet_grid/pred_test_' + str(i) + '.png'
#         print(f't={t}, i={i}')
#         image_filenames.append(image_filename)
#     masks_to_submission(submission_filename, *image_filenames, t)