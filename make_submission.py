from utils.submission import masks_to_submission

submission_filename = './predictions/DBSCAN_-_/netspresso_dbscan_0.75k_submission.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = './predictions/DBSCAN_-_/pred_test_' + str(i) + '.png'
    print(image_filename)
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)