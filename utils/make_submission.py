from submission import masks_to_submission

submission_filename = './predictions/original_thi_DBSCAN/orig_100_16_dbscan.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = './predictions/original_thi_DBSCAN/pred_test_' + str(i) + '.png'
    print(image_filename)
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)