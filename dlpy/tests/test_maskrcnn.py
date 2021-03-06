#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# NOTE: This test requires a running CAS server.  You must use an ~/.authinfo
#       file to specify your username and password.  The CAS host and port must
#       be specified using the CASHOST and CASPORT environment variables.
#       A specific protocol ('cas', 'http', 'https', or 'auto') can be set using
#       the CASPROTOCOL environment variable.
#

import swat
import swat.utils.testing as tm
import unittest
import json
import os
from dlpy.applications import *
from dlpy.applications.application_utils import rpn_layer_options, fast_rcnn_options
from dlpy.layers import RegionProposal, ROIPooling, FastRCNN, ROIAlign, MaskRCNN
from dlpy.model import MomentumSolver, Optimizer, DataSpec, Gpu


# Defined a Toy_FasterRCNN to speed up the test
def Toy_MaskRCNN(conn, model_table='TOY_Faster_RCNN', n_channels=3, width=1000, height=496, scale=1,
                 norm_stds=None, offsets=(102.9801, 115.9465, 122.7717), random_mutation=None,
                 n_classes=20, anchor_num_to_sample=256, anchor_ratio=[0.5, 1, 2], anchor_scale=[8, 16, 32],
                 base_anchor_size=16, coord_type='coco', max_label_per_image=200, proposed_roi_num_train=2000,
                 proposed_roi_num_score=300, roi_train_sample_num=128, roi_pooling_height=7, roi_pooling_width=7,
                 nms_iou_threshold=0.3, detection_threshold=0.5, max_object_num=50, number_of_neurons_in_fc=4096,
                 roialign_height=14, roialign_width=14, mask_threshold=0.5,
                 random_flip=None, random_crop=None):
    num_anchors = len(anchor_ratio) * len(anchor_scale)
    parameters = locals()
    # get parameters of input, rpn, fast_rcnn layer
    input_parameters = get_layer_options(input_layer_options, parameters)
    rpn_parameters = get_layer_options(rpn_layer_options, parameters)
    fast_rcnn_parameters = get_layer_options(fast_rcnn_options, parameters)

    # build a small toy backbone
    inp_tensor = Input(**input_parameters, name='data')
    backbone_conv_out = Conv2d(8, 3, 3, 16, 'conv0')(inp_tensor)

    rpn_conv = Conv2d(width=3, n_filters=512, name='rpn_conv_3x3')(backbone_conv_out)
    rpn_score = Conv2d(act='identity', width=1, n_filters=((1 + 1 + 4) * num_anchors), name='rpn_score')(rpn_conv)
    rpn_out = RegionProposal(**rpn_parameters, name='rois')(rpn_score)
    roipool1 = ROIPooling(output_height=roi_pooling_height, output_width=roi_pooling_width,
                          spatial_scale=backbone_conv_out.shape[0] / height,
                          name='roi_pooling')([backbone_conv_out, rpn_out])
    fc6 = Dense(n=number_of_neurons_in_fc, act='relu', name='fc6')(roipool1)
    fc7 = Dense(n=number_of_neurons_in_fc, act='relu', name='fc7')(fc6)
    cls1 = Dense(n=n_classes + 1, act='identity', name='cls_score')(fc7)
    reg1 = Dense(n=(n_classes + 1) * 4, act='identity', name='bbox_pred')(fc7)
    rcnn_out = FastRCNN(**fast_rcnn_parameters, class_number=n_classes, name='fastrcnn')([cls1, reg1, rpn_out])

    roi_align_out = ROIAlign(output_width=roialign_width, output_height=roialign_height, spatial_scale=backbone_conv_out.shape[0] / width,
                             name='roi_align')([backbone_conv_out, rpn_out])


    prev = Conv2d(act='relu', width=3, height=3, n_filters=256, stride=1, padding=1,
                  name="mask_fcn")(roi_align_out)
    mask_tensor = Conv2d(act='relu', width=1, height=1, n_filters=n_classes+1, stride=1,
                         name="mask_predictor")(prev)
    maskrcnn_out = MaskRCNN(class_number=n_classes, mask_threshold=mask_threshold, name='mask_rcnn')(
        [mask_tensor, rcnn_out, rpn_out])
    mask_rcnn_model = Model(conn, inp_tensor, [rcnn_out, maskrcnn_out], model_table=model_table)
    mask_rcnn_model.compile()
    return mask_rcnn_model

class TestMaskRCNN(unittest.TestCase):
    server_type = None
    s = None
    server_sep = '/'
    data_dir = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False
        self.s = swat.CAS()
        self.server_type = tm.get_cas_host_type(self.s)

        self.server_sep = '\\'
        if self.server_type.startswith("lin") or self.server_type.startswith("osx"):
            self.server_sep = '/'

        if 'DLPY_DATA_DIR' in os.environ:
            self.data_dir = os.environ.get('DLPY_DATA_DIR')
            if self.data_dir.endswith(self.server_sep):
                self.data_dir = self.data_dir[:-1]
            self.data_dir += self.server_sep

        try:
            filename = os.path.join('datasources', 'sample_syntax_for_test.json')
            project_path = os.path.dirname(os.path.abspath(__file__))
            full_filename = os.path.join(project_path, filename)
            with open(full_filename) as f:
                self.sample_syntax = json.load(f)
        except:
            self.sample_syntax = None

    def tearDown(self):
        # tear down tests
        try:
            self.s.terminate()
        except swat.SWATError:
            pass
        del self.s
        swat.reset_option()

    def testsummary(self):
        anchor_ratio = [0.5, 1, 2]
        anchor_scale = [16, 32, 64]
        base_anchor_size = 16
        maskrnn_model = Toy_MaskRCNN(self.s, width=496, height=496, offsets=(103.939, 116.779, 123.68), n_classes=21,
                                     anchor_num_to_sample=256, anchor_ratio=anchor_ratio, anchor_scale=anchor_scale,
                                     base_anchor_size=base_anchor_size, coord_type='coco', proposed_roi_num_score=300,
                                     roi_pooling_height=7, roi_pooling_width=7, nms_iou_threshold=0.1,
                                     detection_threshold=0.5, max_object_num=100, number_of_neurons_in_fc=2048,
                                     roialign_height=14, roialign_width=14, mask_threshold=0.5)
        maskrnn_model.print_summary()

    def test_maskrcnn_missing_spec(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('imageInstantMaskNOclu1000_1.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        #max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())
        max_objs = 6;
        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["xmin", "ymin", "xmax", "ymax"]:
                targets.append('_Object%d_%s' % (i, sp))

        input_vars = []
        input_vars.insert(0, '_image_')
        mask_vars = []
        mask_vars.insert(0, 'labels')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=input_vars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets)
                      ]

        maskrcnn_model = Toy_MaskRCNN(self.s, n_channels=3, n_classes=6)

        res = maskrcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.debug, '0x903fe97d:TKDL_DATASPEC_REQUIRED')

    def test_maskrcnn_wrong_class(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('imageInstantMaskNOclu1000_1.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        #max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())
        max_objs = 6
        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["xmin", "ymin", "xmax", "ymax"]:
                targets.append('_Object%d_%s' % (i, sp))

        input_vars = []
        input_vars.insert(0, '_image_')
        mask_vars = []
        mask_vars.insert(0, 'labels')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=input_vars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets),
                      DataSpec(type_='IMAGE', layer='mask_rcnn', data=mask_vars)
                      ]

        maskrcnn_model = Toy_MaskRCNN(self.s, n_channels=3, n_classes=6)

        res = maskrcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.debug, '0x903fe995:TKDL_OBJDET_CLASSNUM_MISMATCH')

    def test_maskrcnn_train_cpu_normal(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('imageInstantMaskNOclu1000_1.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        #max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())
        max_objs = 6
        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["xmin", "ymin", "xmax", "ymax"]:
                targets.append('_Object%d_%s' % (i, sp))

        input_vars = []
        input_vars.insert(0, '_image_')
        mask_vars = []
        mask_vars.insert(0, 'labels')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=input_vars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets),
                      DataSpec(type_='IMAGE', layer='mask_rcnn', data=mask_vars)
                      ]

        maskrcnn_model = Toy_MaskRCNN(self.s, n_channels=3, n_classes=2)

        res = maskrcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.status_code, 0)
        self.assertEqual(len(res.messages), 31)
        self.assertEqual(len(res.OptIterHistory.Loss), 1)

    def test_maskrcnn_train_freezeLayers (self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('imageInstantMaskNOclu1000_1.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        #max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())
        max_objs = 6
        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["xmin", "ymin", "xmax", "ymax"]:
                targets.append('_Object%d_%s' % (i, sp))

        input_vars = []
        input_vars.insert(0, '_image_')
        mask_vars = []
        mask_vars.insert(0, 'labels')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005,
                              freeze_layers_to='roi_align')
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=input_vars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets),
                      DataSpec(type_='IMAGE', layer='mask_rcnn', data=mask_vars)
                      ]

        maskrcnn_model = Toy_MaskRCNN(self.s, n_channels=3, n_classes=2)

        res = maskrcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.status_code, 0)
        self.assertEqual(len(res.messages), 31)
        self.assertEqual(len(res.OptIterHistory.Loss), 1)

        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005,
                              freeze_layers_to='mask_rcnn')

        res = maskrcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.status_code, 0)
        self.assertEqual(len(res.messages), 31)
        self.assertEqual(len(res.OptIterHistory.Loss), 1)

    def test_maskrcnn_score (self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('imageInstantMaskNOclu1000_1.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        #max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())
        max_objs = 6
        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["xmin", "ymin", "xmax", "ymax"]:
                targets.append('_Object%d_%s' % (i, sp))

        input_vars = []
        input_vars.insert(0, '_image_')
        mask_vars = []
        mask_vars.insert(0, 'labels')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=input_vars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets),
                      DataSpec(type_='IMAGE', layer='mask_rcnn', data=mask_vars)
                      ]

        maskrcnn_model = Toy_MaskRCNN(self.s, n_channels=3, n_classes=2)

        res = maskrcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       gpu=Gpu(devices=0),
                       force_equal_padding=True)
        self.assertEqual(res.status_code, 0)
        self.assertEqual(len(res.messages), 32)
        self.assertEqual(len(res.OptIterHistory.Loss), 1)

        res_score = maskrcnn_model.predict(data='trainset', n_threads=1)
        self.assertEqual(res.status_code, 0)
        self.assertEqual(res_score.ScoreInfo.size, 8)

if __name__ == '__main__':
    unittest.main()
