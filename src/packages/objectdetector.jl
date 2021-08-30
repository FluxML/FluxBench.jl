function objectdetector_add_yolo_fw(model = YOLO.v3_608_COCO, batchsize = 1, od_group = addgroup!(SUITE, "ObjectDetector"))
  yolomod = model(batch=batchsize, silent=true) # Load the YOLOv3-tiny model pretrained on COCO, with a batch size of 1
  batch = emptybatch(yolomod) # Create a batch object. Automatically uses the GPU if available
  img = load(joinpath(pkgdir(ObjectDetector),"test","images","dog-cycle-car.png"))
  img_resize, padding = prepareImage(img, yolomod) # Send resized image to the batch
  for i in 1:batchsize
    batch[:,:,:,i] = img_resize # Send resized image to the batch
  end

  od_group["ObjectDetector_$(model)_with_batchsize_$(batchsize)"] = b = @benchmarkable(
    yolomod(batch, detectThresh=0.5, overlapThresh=0.8),
    setup = (yolomod = gpu($yolomod);
             batch = gpu($batch)),
    teardown=(GC.gc(); CUDA.reclaim()))
end
