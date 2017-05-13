//
//  ViewController.swift
//  GPUExample
//
//  Created by Mateusz Buda on 12/04/15.
//  Copyright (c) 2015 inFullMobile. All rights reserved.
//

import UIKit
import Metal
import QuartzCore
import Darwin
import Accelerate

let PROBLEM_SIZE = 16777216 // 2^24
let RESULT_SIZE = 1
let THREADGROUP_SIZE = 256

class ViewController: UIViewController {

    var kernelName: String!

    @IBOutlet weak var execTimeGPU: UILabel!
    @IBOutlet weak var execTimeCPU: UILabel!

    var input: [Int32] = [Int32](repeating: 0, count: PROBLEM_SIZE)
    var result: [Int32] = [Int32](repeating: 0, count: RESULT_SIZE)

    override func viewDidLoad() {
        super.viewDidLoad()

        title = kernelName
    }

    @IBAction func runGPU(_ sender: UIButton) {
        let (device, _, defaultLibrary, commandBuffer, computeCommandEncoder) = initMetal()

        let resultSize = kernelName == "map" ? PROBLEM_SIZE : RESULT_SIZE
        input = [Int32](repeating: (kernelName == "map" ? 0 : 1), count: PROBLEM_SIZE)
        result = [Int32](repeating: 0, count: resultSize)

        // set up a compute pipeline with kernel function and add it to encoder
        let kernel = defaultLibrary.makeFunction(name: kernelName)


        do {
            let computePipelineState = try device.makeComputePipelineState(function: kernel!)
            computeCommandEncoder.setComputePipelineState(computePipelineState)
        } catch {
            computeCommandEncoder.endEncoding()
            return
        }



        // calculate byte length of input and output data
        let inputByteLength = input.count * MemoryLayout.size(ofValue: input[0])
        var resultByteLength = result.count * MemoryLayout.size(ofValue: result[0])

        // create a MTLBuffer - input data for GPU (<= 256 MB)
        let inputBuffer = device.makeBuffer(bytes: &input, length: inputByteLength, options: [])

        // set the input vector for the kernel function,
        // atIndex: 0 here corresponds to buffer(0) in the kernel function
        computeCommandEncoder.setBuffer(inputBuffer, offset: 0, at: 0)

        // create the output buffer for the kernel function,
        // atIndex: 1 here corresponds to buffer(1) in the kernel function
        let resultBuffer = device.makeBuffer(bytes: &result, length: resultByteLength, options: [])
        computeCommandEncoder.setBuffer(resultBuffer, offset: 0, at: 1)

        // make grid
        let threadgroupSizeMultiplier = kernelName.contains("4") ? 2 : 1
        let threadsPerGroup = MTLSize(width: THREADGROUP_SIZE, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (PROBLEM_SIZE / (THREADGROUP_SIZE * threadgroupSizeMultiplier)), height: 1, depth:1)

        print("Block: \(threadsPerGroup.width) x \(threadsPerGroup.height)\n" +
            "Grid: \(numThreadgroups.width) x \(numThreadgroups.height) x \(numThreadgroups.depth)")

        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)

        // compute and wait for result
        computeCommandEncoder.endEncoding()

        let start = CACurrentMediaTime()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let stop = CACurrentMediaTime()

        if (commandBuffer.error != nil) {
            execTimeGPU.text = "error"
            print("Command buffer error: \(String(describing: commandBuffer.error))")
            return
        }

        execTimeGPU.text = String.localizedStringWithFormat("%.2f ms", (stop-start) * 1000)

        // Get GPU data

//        var data = Data(bytesNoCopy: resultBuffer.contents(), count: resultByteLength, deallocator: )

        let data = Data(bytesNoCopy: resultBuffer.contents(), count: resultByteLength, deallocator: .none)
        // get data from GPU into Swift array
        data.getBytes(&result, length: resultByteLength)

//        print("result = \(result[0])")
    }

    @IBAction func runCPU(_ sender: AnyObject) {
        let resultSize = kernelName == "map" ? PROBLEM_SIZE : RESULT_SIZE
        input = [Int32](repeating: (kernelName == "map" ? 0 : 1), count: PROBLEM_SIZE)
        result = [Int32](repeating: 0, count: resultSize)

        let start = CACurrentMediaTime()

        if kernelName == "map" {

            for i in 0 ..< input.count {
                result[i] = Int32(cos(CDouble(input[i])))
            }

        } else { // reduce

            for i in 0 ..< input.count {
                result[0] += input[i]
            }

        }

        let stop = CACurrentMediaTime()

        execTimeCPU.text = String.localizedStringWithFormat("%.2f ms", (stop-start) * 1000)

        print("result = \(result[0])")
    }

    // MARK: - Metal

    // source:
    // DATA-PARALLEL PROGRAMMING WITH METAL AND SWIFT FOR IPHONE/IPAD GPU by Amund Tveit
    // http://memkite.com/blog/2014/12/15/data-parallel-programming-with-metal-and-swift-for-iphoneipad-gpu/
    func initMetal() -> (MTLDevice, MTLCommandQueue, MTLLibrary, MTLCommandBuffer, MTLComputeCommandEncoder) {
        // Get access to iPhone or iPad GPU
        let device = MTLCreateSystemDefaultDevice()

        // Queue to handle an ordered list of command buffers
        let commandQueue = device!.makeCommandQueue()

        // Access to Metal functions that are stored in Kernel.metal file, e.g. reduce()
        let defaultLibrary = device!.newDefaultLibrary()

        // Buffer for storing encoded commands that are sent to GPU
        let commandBuffer = commandQueue.makeCommandBuffer()

        // Encoder for GPU commands
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()

        return (device!, commandQueue, defaultLibrary!, commandBuffer, computeCommandEncoder)
    }
    
}

