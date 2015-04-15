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

let PROBLEM_SIZE = 63999999 // 256 MB - 4 B
let RESULT_SIZE = 1
let THREADGROUP_SIZE = 512

class ViewController: UIViewController {
    @IBOutlet weak var resultLabel: UILabel!
    
    var input: [Int32] = [Int32](count: PROBLEM_SIZE, repeatedValue: 1)
    var result: [Int32] = [Int32](count:RESULT_SIZE, repeatedValue: 0)

    override func viewDidLoad() {
        super.viewDidLoad()
    }

    @IBAction func runGPU(sender: UIButton) {
        var (device, commandQueue, defaultLibrary, commandBuffer, computeCommandEncoder) = initMetal()
        
        result = [Int32](count:RESULT_SIZE, repeatedValue: 0)
        
        // set up a compute pipeline with sumKernel function and add it to encoder
        let reduceKernel = defaultLibrary.newFunctionWithName("reduce4")
        var pipelineErrors: NSError?
        var computePipelineState = device.newComputePipelineStateWithFunction(reduceKernel!, error: &pipelineErrors)
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
        
        // set the input vector for the reduce function,
        // atIndex: 0 here corresponds to buffer(0) in the reduce function
        computeCommandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
        
        // create the output buffer for the reduce function,
        // atIndex: 1 here corresponds to buffer(1) in the reduce function
        var resultBuffer = device.newBufferWithBytes(&result, length: resultByteLength, options: nil)
        computeCommandEncoder.setBuffer(resultBuffer, offset: 0, atIndex: 1)
        
        // make grid
        var threadsPerGroup = MTLSize(width: THREADGROUP_SIZE, height: 1, depth: 1)
        var numThreadgroups = MTLSize(width: (PROBLEM_SIZE / (THREADGROUP_SIZE * 2)) + 1, height: 1, depth:1)
        
        println("Block: \(threadsPerGroup.width) x \(threadsPerGroup.height)\nGrid: \(numThreadgroups.width) x \(numThreadgroups.height) x \(numThreadgroups.depth)")
        
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        // compute and wait for result
        computeCommandEncoder.endEncoding()
        
        let start = CACurrentMediaTime()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let stop = CACurrentMediaTime()
        println("GPU exec time: \((stop-start) * 1000) milis")
        
        if (commandBuffer.error != nil) {
            println("Command buffer error: \(commandBuffer.error?.debugDescription)")
        }
        
        // Get GPU data
        var data = NSData(bytesNoCopy: resultBuffer.contents(), length: resultByteLength, freeWhenDone: false)
        
        // get data from GPU into Swift array
        data.getBytes(&result, length: resultByteLength)
        
        resultLabel.text = "\(result[0])"
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
    
    @IBAction func runCPU(sender: AnyObject) {
        result = [Int32](count:RESULT_SIZE, repeatedValue: 0)
        
        let start = CACurrentMediaTime()
        
        for i in input {
            result[0] += i
        }
        
        let stop = CACurrentMediaTime()
        println("CPU exec time: \((stop-start) * 1000) milis")
        
        resultLabel.text = "\(result[0])"
    }
}

