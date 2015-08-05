//
//  KernelSelectionController.swift
//  GPUExample
//
//  Created by Mateusz Buda on 05/08/15.
//  Copyright (c) 2015 Antipattern. All rights reserved.
//

import UIKit

class KernelSelectionController: UITableViewController {
    
    let reusableIdentifier = "cell"
    
    var selectedKernel: String!
    
    // MARK: - Delegate
    
    override func tableView(tableView: UITableView, willSelectRowAtIndexPath indexPath: NSIndexPath) -> NSIndexPath? {
        selectedKernel = indexPath.row > 0 ? "reduce\(indexPath.row)" : "map"
        
        return indexPath
    }

    // MARK: - Navigation

    override func prepareForSegue(segue: UIStoryboardSegue, sender: AnyObject?) {
        let destController: ViewController = segue.destinationViewController as! ViewController
        destController.kernelName = selectedKernel
    }

}
