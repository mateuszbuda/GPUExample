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

let PROBLEM_SIZE = 33292288 // 127 MB (the memory limit on device is 256MB, but for map we need space for input and output array)
let RESULT_SIZE = 1
let THREADGROUP_SIZE = 256

class ViewController: UIViewController {
    
    var kernelName: String!
    
    @IBOutlet weak var execTimeGPU: UILabel!
    @IBOutlet weak var execTimeCPU: UILabel!
    
    var input: [Int32] = [Int32](count: PROBLEM_SIZE, repeatedValue: 0)
    var result: [Int32] = [Int32](count: RESULT_SIZE, repeatedValue: 0)
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        title = kernelName
    }

    @IBAction func runGPU(sender: UIButton) {
        var (device, commandQueue, defaultLibrary, commandBuffer, computeCommandEncoder) = initMetal()
        
        let resultSize = kernelName == "map" ? PROBLEM_SIZE : RESULT_SIZE
        result = [Int32](count: resultSize, repeatedValue: 0)
        
        // set up a compute pipeline with kernel function and add it to encoder
        let kernel = defaultLibrary.newFunctionWithName(kernelName)
        var pipelineErrors: NSError?
        var computePipelineState = device.newComputePipelineStateWithFunction(kernel!, error: &pipelineErrors)
        if computePipelineState == nil {
            println("Failed to create pipeline state, error: \(pipelineErrors?.debugDescription)")
            computeCommandEncoder.endEncoding()
            return
        }
        computeCommandEncoder.setComputePipelineState(computePipelineState!)
        
        // calculate byte length of input and output data
        var inputByteLength = input.count * sizeofValue(input[0])
        var resultByteLength = result.count * sizeofValue(result[0])
        
        // create a MTLBuffer - input data for GPU (<= 256 MB)
        var inputBuffer = device.newBufferWithBytes(&input, length: inputByteLength, options: nil)
        
        // set the input vector for the kernel function,
        // atIndex: 0 here corresponds to buffer(0) in the kernel function
        computeCommandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
        
        // create the output buffer for the kernel function,
        // atIndex: 1 here corresponds to buffer(1) in the kernel function
        var resultBuffer = device.newBufferWithBytes(&result, length: resultByteLength, options: nil)
        computeCommandEncoder.setBuffer(resultBuffer, offset: 0, atIndex: 1)
        
        // make grid
        let threadgroupSizeMultiplier = contains(kernelName, "4") ? 2 : 1
        var threadsPerGroup = MTLSize(width: THREADGROUP_SIZE, height: 1, depth: 1)
        var numThreadgroups = MTLSize(width: (PROBLEM_SIZE / (THREADGROUP_SIZE * threadgroupSizeMultiplier)) + 1, height: 1, depth:1)
        
        println("Block: \(threadsPerGroup.width) x \(threadsPerGroup.height)\n" +
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
            println("Command buffer error: \(commandBuffer.error?.debugDescription)")
            return
        }
        
        execTimeGPU.text = String.localizedStringWithFormat("%.2f ms", (stop-start) * 1000)
        
        // Get GPU data
        var data = NSData(bytesNoCopy: resultBuffer.contents(), length: resultByteLength, freeWhenDone: false)
        
        // get data from GPU into Swift array
        data.getBytes(&result, length: resultByteLength)
        
        println("result = \(result[0])")
    }
    
    @IBAction func runCPU(sender: AnyObject) {
        let resultSize = kernelName == "map" ? PROBLEM_SIZE : RESULT_SIZE
        result = [Int32](count: resultSize, repeatedValue: 0)
        
        let start = CACurrentMediaTime()
        
        if kernelName == "map" {
            
            for i in 0 ..< input.count {
                result[i] = Int32(cos(CDouble(input[i])))
            }
            
        } else { // reduce
            
            for i in input {
                result[0] += Int32(cos(CDouble(i)))
            }
            
        }
        
        let stop = CACurrentMediaTime()
        
        execTimeCPU.text = String.localizedStringWithFormat("%.2f ms", (stop-start) * 1000)
        
        println("result = \(result[0])")
    }
    
    // MARK: - Metal
    
    func initMetal() -> (MTLDevice, MTLCommandQueue, MTLLibrary, MTLCommandBuffer, MTLComputeCommandEncoder) {
        // Get access to iPhone or iPad GPU
        var device = MTLCreateSystemDefaultDevice()
        
        // Queue to handle an ordered list of command buffers
        var commandQueue = device.newCommandQueue()
        
        // Access to Metal functions that are stored in Kernel.metal file, e.g. reduce()
        var defaultLibrary = device.newDefaultLibrary()
        
        // Buffer for storing encoded commands that are sent to GPU
        var commandBuffer = commandQueue.commandBuffer()
        
        // Encoder for GPU commands
        var computeCommandEncoder = commandBuffer.computeCommandEncoder()
        
        return (device, commandQueue, defaultLibrary!, commandBuffer, computeCommandEncoder)
    }
    
}

