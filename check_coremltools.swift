#!/usr/bin/env swift
/// Tries to load MNISTClassifier.mlmodel from the present directory
/// Tried to use it to run inference on the image passed as the first argument
import Foundation
import CoreML
import Vision

/// Synchronously executes inference on image using a model
func infer(modelPath:String, imagePath:String) -> Bool
{
  let sema = DispatchSemaphore.init(value: 0)

  func processClassification(for request:VNCoreMLRequest, error:Error?)
  {
    print("processClassifacation() called")
    defer { sema.signal() }
    if let e = error {
      print("Error processing classification request: \(e)")
      return
    }

    guard let results = request.results as? [VNClassificationObservation] else {
      print("No results")
      return
    }

    if results.isEmpty {
      print("no results")
    }
    else {
      for c in results {
        print(c)
      }
    }
  }

  let modelURL:URL = URL(fileURLWithPath: modelPath).standardizedFileURL
  guard FileManager.default.fileExists(atPath:modelURL.path) else {
    print("model file does not exist at \(modelURL) which has path \(modelURL.path)")
    return false
  }

  let model:MLModel
  do {
    let compiledModelURL = try MLModel.compileModel(at:modelURL)

    model = try MLModel(contentsOf: compiledModelURL,
                        configuration: MLModelConfiguration())
  }
  catch let e {
    print("Could not initialize an MLModel from \(modelPath).")
    print("Got this error: \(e)")
    return false
  }

  let imageURL = URL(fileURLWithPath: imagePath)

  let the_request:VNCoreMLRequest
  do {
    let model = try VNCoreMLModel(for: model)

    the_request = VNCoreMLRequest(model: model, completionHandler: { (request:VNRequest, error:Error?) in
      processClassification(for: request as! VNCoreMLRequest, error: error)
    })
    the_request.imageCropAndScaleOption = .centerCrop
  } catch {
    print("Failed to load Vision ML model: \(error)")
    return false
  }

  let requestHandler = VNImageRequestHandler(url: imageURL, options: [:])
  do {
    try requestHandler.perform([the_request])
  }
  catch let e {
    print("Error trying to perform the classification request: \(e.localizedDescription)")
  }

  let result = sema.wait(timeout: DispatchTime.now().advanced(by: DispatchTimeInterval.seconds(5)))

  if result != DispatchTimeoutResult.success {
    print("Timed out waiting for processing")
  }

  return true
}


_ = infer(modelPath:"./MNISTClassifier.mlmodel",
      imagePath:CommandLine.arguments[1])

print("infer returned")

