//
//  KernelSelectionController.swift
//  GPUExample
//
//  Created by Mateusz Buda on 05/08/15.
//  Copyright (c) 2015 inFullMobile. All rights reserved.
//

import UIKit

class KernelSelectionController: UITableViewController {
    
    var selectedKernel: String!
    
    // MARK: - Delegate
    
    override func tableView(_ tableView: UITableView, willSelectRowAt indexPath: IndexPath) -> IndexPath? {
        selectedKernel = indexPath.row > 0 ? "reduce\(indexPath.row)" : "map"
        
        return indexPath
    }

    // MARK: - Navigation

    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        let destController: ViewController = segue.destination as! ViewController
        destController.kernelName = selectedKernel
    }

}
