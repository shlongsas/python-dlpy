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
from dlpy.layers import RegionProposal, ROIPooling, FastRCNN
from dlpy.model import MomentumSolver, Optimizer, DataSpec, Gpu

# Defined a Toy_FasterRCNN to speed up the test
def Toy_FasterRCNN (conn, model_table='TOY_Faster_RCNN', n_channels=3, width=1000, height=496, scale=1,
                norm_stds=None, offsets=(102.9801, 115.9465, 122.7717), random_mutation=None,
                n_classes=20, anchor_num_to_sample=256, anchor_ratio=[0.5, 1, 2], anchor_scale=[8, 16, 32],
                base_anchor_size=16, coord_type='coco', max_label_per_image=200, proposed_roi_num_train=2000,
                proposed_roi_num_score=300, roi_train_sample_num=128, roi_pooling_height=7, roi_pooling_width=7,
                nms_iou_threshold=0.3, detection_threshold=0.5, max_object_num=50, number_of_neurons_in_fc=4096,
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
    rp1 = RegionProposal(**rpn_parameters, name='rois')(rpn_score)
    roipool1 = ROIPooling(output_height=roi_pooling_height, output_width=roi_pooling_width,
                          spatial_scale=backbone_conv_out.shape[0] / height,
                          name='roi_pooling')([backbone_conv_out, rp1])
    fc6 = Dense(n=number_of_neurons_in_fc, act='relu', name='fc6')(roipool1)
    fc7 = Dense(n=number_of_neurons_in_fc, act='relu', name='fc7')(fc6)
    cls1 = Dense(n=n_classes + 1, act='identity', name='cls_score')(fc7)
    reg1 = Dense(n=(n_classes + 1) * 4, act='identity', name='bbox_pred')(fc7)
    fr1 = FastRCNN(**fast_rcnn_parameters, class_number=n_classes, name='fastrcnn')([cls1, reg1, rp1])
    faster_rcnn_model = Model(conn, inp_tensor, fr1, model_table=model_table)
    faster_rcnn_model.compile()
    return faster_rcnn_model

class TestDetection(unittest.TestCase):
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

    def test_YoloV2_train_classnum_mismatch(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('chess_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        #yolo_anchors = get_anchors(self.s, coord_type='yolo', data='trainset')
        yolo_anchors = [
            2696.873493975904,
            1078.1472556894244,
            3283.62248995984,
            2736.949576082106,
            2045.5617469879517,
            3366.982597054886,
            2347.092943201377,
            2183.3436603557084,
            2782.939759036145,
            4885.9330655957165
        ]
        yolo_model = Tiny_YoloV2(self.s,
                                 n_classes=2,
                                 predictions_per_grid=5,
                                 anchors=yolo_anchors,
                                 max_boxes=100,
                                 coord_type='yolo',
                                 max_label_per_image=1,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)

        targets = ['_nObjects_']
        for i in range(0, 3):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='Input1', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='Detection1', data=targets)]
        res = yolo_model.fit(data='trainset',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual (res.debug, '0x903fe995:TKDL_OBJDET_CLASSNUM_MISMATCH')

    def test_YoloV2_train_objs_exceed_maximum(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('chess_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        #yolo_anchors = get_anchors(self.s, coord_type='yolo', data='trainset')
        yolo_anchors = [
            2696.873493975904,
            1078.1472556894244,
            3283.62248995984,
            2736.949576082106,
            2045.5617469879517,
            3366.982597054886,
            2347.092943201377,
            2183.3436603557084,
            2782.939759036145,
            4885.9330655957165
        ]
        yolo_model = Tiny_YoloV2(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 anchors=yolo_anchors,
                                 max_boxes=100,
                                 coord_type='yolo',
                                 max_label_per_image=1,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)

        targets = ['_nObjects_']
        for i in range(0, 3):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='Input1', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='Detection1', data=targets)]
        res = yolo_model.fit(data='trainset',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual (res.debug, '0x903fea0c:TKDL_DETECTION_LABELED_OBJECTS_EXCEED_MAXIMUM')

    def test_YoloV2_train_coord_mismatch(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('chess_coco.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        #yolo_anchors = get_anchors(self.s, coord_type='yolo', data='trainset')
        yolo_anchors = [
            2696.873493975904,
            1078.1472556894244,
            3283.62248995984,
            2736.949576082106,
            2045.5617469879517,
            3366.982597054886,
            2347.092943201377,
            2183.3436603557084,
            2782.939759036145,
            4885.9330655957165
        ]
        yolo_model = Tiny_YoloV2(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 anchors=yolo_anchors,
                                 max_boxes=100,
                                 coord_type='yolo',
                                 max_label_per_image=1,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)

        targets = ['_nObjects_']
        for i in range(0, 3):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='Input1', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='Detection1', data=targets)]
        res = yolo_model.fit(data='trainset',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual (res.debug, '0x8affc037:TKCASU_COLUMN_DOESNOT_EXIST')

        self.s.loadtable('chess_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset_yolo', replace=1))
        yolo_model = Tiny_YoloV2(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 anchors=yolo_anchors,
                                 max_boxes=100,
                                 coord_type='rect',
                                 max_label_per_image=1000,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)
        res = yolo_model.fit(data='trainset_yolo',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual (res.debug, '0x903feb15:TKDL_LABELED_OBJECT_UNUSUAL')

        soptimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=1, reg_l2=0.005)
        optimizer.__setitem__('ignore_training_error', True)
        res = yolo_model.fit(data='trainset_yolo',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       log_level=5,
                       force_equal_padding=True)
        self.assertEqual(res.status_code, 0)
        self.assertEqual(len(res.messages), 18)
        self.assertEqual(len(res.OptIterHistory.Loss), 1)

    def test_YoloV2_train_nodataspec(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('chess_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        yolo_anchors = [
            2696.873493975904,
            1078.1472556894244,
            3283.62248995984,
            2736.949576082106,
            2045.5617469879517,
            3366.982597054886,
            2347.092943201377,
            2183.3436603557084,
            2782.939759036145,
            4885.9330655957165
        ]
        yolo_model = Tiny_YoloV2(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 anchors=yolo_anchors,
                                 max_boxes=100,
                                 coord_type='yolo',
                                 max_label_per_image=100,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)

        targets = ['_nObjects_']
        for i in range(0, 3):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=2, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='Input1', data=inputVars)]
        res = yolo_model.fit(data='trainset',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       force_equal_padding=True)

        self.assertEqual(res.debug, '0x903fe97d:TKDL_DATASPEC_REQUIRED')

    def test_YoloV2_train_cpu_normal(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('chess_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        yolo_anchors = [
            2696.873493975904,
            1078.1472556894244,
            3283.62248995984,
            2736.949576082106,
            2045.5617469879517,
            3366.982597054886,
            2347.092943201377,
            2183.3436603557084,
            2782.939759036145,
            4885.9330655957165
        ]
        yolo_model = Tiny_YoloV2(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 anchors=yolo_anchors,
                                 max_boxes=100,
                                 coord_type='yolo',
                                 max_label_per_image=100,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)

        targets = ['_nObjects_']
        for i in range(0, 3):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=2, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='Input1', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='Detection1', data=targets)]
        res = yolo_model.fit(data='trainset',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       force_equal_padding=True)

        self.assertEqual(res.status_code, 0)
        self.assertEqual(len(res.messages), 19)
        self.assertEqual(len(res.OptIterHistory.Loss), 2)

    def test_YoloV2_train_gpu_normal(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('chess_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        yolo_anchors = [
            2696.873493975904,
            1078.1472556894244,
            3283.62248995984,
            2736.949576082106,
            2045.5617469879517,
            3366.982597054886,
            2347.092943201377,
            2183.3436603557084,
            2782.939759036145,
            4885.9330655957165
        ]
        yolo_model = Tiny_YoloV2(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 anchors=yolo_anchors,
                                 max_boxes=100,
                                 coord_type='yolo',
                                 max_label_per_image=100,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)

        targets = ['_nObjects_']
        for i in range(0, 3):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=2, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='Input1', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='Detection1', data=targets)]

        res_gpu = yolo_model.fit(data='trainset',
                             optimizer=optimizer,
                             data_specs=data_specs,
                             n_threads=2,
                             record_seed=13309,
                             gpu=Gpu(devices=0),
                             force_equal_padding=True)

        self.assertEqual(res_gpu.status_code, 0)
        self.assertEqual(len(res_gpu.messages), 21)
        self.assertEqual(len(res_gpu.OptIterHistory.Loss), 2)

    def test_YoloV1_train (self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('chess_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        yolo_model = Tiny_YoloV1(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 max_boxes=100,
                                 coord_type='yolo',
                                 max_label_per_image=100,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)

        targets = ['_nObjects_']
        for i in range(0, 3):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=2, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='Input1', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='Detection1', data=targets)]
        res = yolo_model.fit(data='trainset',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       force_equal_padding=True)

        self.assertEqual(res.debug, '0x903fe880:TKDL_IMAGEINFO_WRONG')
        yolo_model = Tiny_YoloV1(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 max_boxes=100,
                                 coord_type='yolo',
                                 max_label_per_image=100,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 width=416,
                                 height=416,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)
        res = yolo_model.fit(data='trainset',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.status_code, 0)
        self.assertEqual(len(res.messages), 20)
        self.assertEqual(len(res.OptIterHistory.Loss), 2)

    def test_YoloV2_score(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('chess_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='testset', replace=1))

        yolo_anchors = [
            2696.873493975904,
            1078.1472556894244,
            3283.62248995984,
            2736.949576082106,
            2045.5617469879517,
            3366.982597054886,
            2347.092943201377,
            2183.3436603557084,
            2782.939759036145,
            4885.9330655957165
        ]
        yolo_model = Tiny_YoloV2(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 anchors=yolo_anchors,
                                 max_boxes=100,
                                 coord_type='yolo',
                                 max_label_per_image=100,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)

        targets = ['_nObjects_']
        for i in range(0, 3):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=3, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='Input1', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='Detection1', data=targets)]
        res_train = yolo_model.fit(data='testset',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       seed=39998,
                       force_equal_padding=True)

        res_score = yolo_model.predict(data='testset', n_threads=2)
        self.assertEqual(res_score.status_code, 0)
        self.assertEqual(res_score.ScoreInfo.size, 8)
        self.assertEqual(len(res_score.messages), 1)

    def test_YoloV2_train_coco(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('chess_coco.sashdat',
                          caslib='dnfs',
                          casout=dict(name='testset', replace=1))

        yolo_anchors = [
            2696.873493975904,
            1078.1472556894244,
            3283.62248995984,
            2736.949576082106,
            2045.5617469879517,
            3366.982597054886,
            2347.092943201377,
            2183.3436603557084,
            2782.939759036145,
            4885.9330655957165
        ]
        yolo_model = Tiny_YoloV2(self.s,
                                 n_classes=4,
                                 predictions_per_grid=5,
                                 anchors=yolo_anchors,
                                 max_boxes=100,
                                 coord_type='coco',
                                 max_label_per_image=100,
                                 class_scale=1.0,
                                 coord_scale=1.0,
                                 prediction_not_a_object_scale=1,
                                 object_scale=5,
                                 detection_threshold=0.2,
                                 iou_threshold=0.2)

        targets = ['_nObjects_']
        for i in range(0, 3):
            targets.append('_Object%d_' % i)
            for sp in ["xmin", "ymin", "xmax", "ymax"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='Input1', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='Detection1', data=targets)]
        res_train = yolo_model.fit(data='testset',
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res_train.status_code, 0)
        self.assertEqual(len(res_train.OptIterHistory.Loss), 1)
        self.assertEqual(len(res_train.messages), 19)


    def test_FasterRCNN_train_classnum_mismatch(self):
        from dlpy.applications import Faster_RCNN

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('rcnn_1000x496_small_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        anchor_ratio = [0.5, 1, 2]
        anchor_scale = [16, 32, 64]
        base_anchor_size = 16
        faster_rcnn_model = Toy_FasterRCNN(self.s, width=496, height=496, offsets=(103.939, 116.779, 123.68), n_classes=21,
                            anchor_num_to_sample=256, anchor_ratio=anchor_ratio, anchor_scale=anchor_scale,
                            base_anchor_size=base_anchor_size, coord_type='coco', proposed_roi_num_score=300,
                            roi_pooling_height=14, roi_pooling_width=14, nms_iou_threshold=0.1, detection_threshold=0.5,
                            max_object_num=100, number_of_neurons_in_fc=2048)

        train_table = self.s.CASTable('trainset')
        max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())

        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=10, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='input', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets)]
        res = faster_rcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual (res.debug, '0x903fe832:TKDL_LAYER_DOESNOT_EXIST')

        data_specs = [DataSpec(type_='IMAGE', layer='data', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets)]
        res = faster_rcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=1,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.debug, '0x903fe995:TKDL_OBJDET_CLASSNUM_MISMATCH')

    def test_FasterRCNN_train_coord_mismatch(self):
        from dlpy.applications import Faster_RCNN

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('rcnn_1000x496_small_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())

        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets)]

        faster_rcnn_model = Toy_FasterRCNN(self.s, n_channels=3, n_classes=6)
        res = faster_rcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.debug, '0x903feb15:TKDL_LABELED_OBJECT_UNUSUAL')

    def test_FasterRCNN_train_nodataspec(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                          datasource={'srctype': 'path'},
                          name='dnfs',
                          path=self.data_dir,
                          subdirectories=False)

        self.s.loadtable('rcnn_1000x496_small_yolo.sashdat',
                          caslib='dnfs',
                          casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())

        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=inputVars)]

        faster_rcnn_model = Toy_FasterRCNN(self.s, n_channels=3, n_classes=6)
        res = faster_rcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.debug, '0x903fe97d:TKDL_DATASPEC_REQUIRED')

    def test_FasterRCNN_train_cpu_normal(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                               datasource={'srctype': 'path'},
                               name='dnfs',
                               path=self.data_dir,
                               subdirectories=False)

        self.s.loadtable('rcnn_1000x496_small_yolo.sashdat',
                         caslib='dnfs',
                         casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())

        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets)]
        faster_rcnn_model = Toy_FasterRCNN(self.s, n_channels=3, n_classes=6, coord_type='Yolo')
        res = faster_rcnn_model.fit(train_table,
                                    optimizer=optimizer,
                                    data_specs=data_specs,
                                    n_threads=2,
                                    record_seed=13309,
                                    force_equal_padding=True)
        self.assertEqual(res.status_code, 0)
        self.assertEqual(len(res.messages), 31)
        self.assertEqual(len(res.OptIterHistory.Loss), 1)

    def test_FasterRCNN_train_gpu_normal(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                               datasource={'srctype': 'path'},
                               name='dnfs',
                               path=self.data_dir,
                               subdirectories=False)

        self.s.loadtable('rcnn_1000x496_small_yolo.sashdat',
                         caslib='dnfs',
                         casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())

        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=2, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets)]
        faster_rcnn_model = Toy_FasterRCNN(self.s, n_channels=3, n_classes=6, coord_type='Yolo')
        res_gpu = faster_rcnn_model.fit(train_table,
                                    optimizer=optimizer,
                                    data_specs=data_specs,
                                    n_threads=2,
                                    record_seed=13309,
                                    gpu=Gpu(devices=0),
                                    force_equal_padding=True)

        self.assertEqual(res_gpu.status_code, 0)
        self.assertEqual(len(res_gpu.OptIterHistory.Loss), 2)
        self.assertEqual(len(res_gpu.messages), 33)

    def test_FasterRCNN_score(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                               datasource={'srctype': 'path'},
                               name='dnfs',
                               path=self.data_dir,
                               subdirectories=False)

        self.s.loadtable('rcnn_1000x496_small_yolo.sashdat',
                         caslib='dnfs',
                         casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        max_objs = 22
        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["x", "y", "width", "height"]:
                targets.append('_Object%d_%s' % (i, sp))


        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets)]
        faster_rcnn_model = Toy_FasterRCNN(self.s, n_channels=3, n_classes=6, coord_type='Yolo')
        res = faster_rcnn_model.fit(train_table,
                                    optimizer=optimizer,
                                    data_specs=data_specs,
                                    n_threads=2,
                                    record_seed=13309,
                                    force_equal_padding=True)

        res_score = faster_rcnn_model.predict(data='trainset', n_threads=2)
        self.assertEqual(res_score.debug, '0x887ff93d:TKCASTAB_DUP_COLUMN')

        for i in range(0, max_objs):
            targets.append('_P_Object%d_' % i)
        for col in targets:
            self.s.altertable(self.s.CASTable("trainset"), columns=[dict(name=col, drop=True)])

        res_score = faster_rcnn_model.predict(data='trainset', n_threads=2)

        self.assertEqual(res_score.status_code, 0)
        self.assertEqual(res_score.ScoreInfo.size, 4)
        self.assertEqual(len(res_score.messages), 1)

    def test_FasterRCNN_train_coco(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                               datasource={'srctype': 'path'},
                               name='dnfs',
                               path=self.data_dir,
                               subdirectories=False)

        self.s.loadtable('rcnn_1000x496_small_coco.sashdat',
                         caslib='dnfs',
                         casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())

        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["xmin", "ymin", "xmax", "ymax"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005)
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets)]
        faster_rcnn_model = Toy_FasterRCNN(self.s, n_channels=3, n_classes=6, coord_type='Coco')
        res_gpu = faster_rcnn_model.fit(train_table,
                                    optimizer=optimizer,
                                    data_specs=data_specs,
                                    n_threads=2,
                                    record_seed=13309,
                                    force_equal_padding=True)

        self.assertEqual(res_gpu.status_code, 0)
        self.assertEqual(len(res_gpu.OptIterHistory.Loss), 1)
        self.assertEqual(len(res_gpu.messages), 31)

    def test_FasterRCNN_train_freeze_layers(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.s.table.addcaslib(activeonadd=False,
                               datasource={'srctype': 'path'},
                               name='dnfs',
                               path=self.data_dir,
                               subdirectories=False)

        self.s.loadtable('rcnn_1000x496_small_coco.sashdat',
                         caslib='dnfs',
                         casout=dict(name='trainset', replace=1))

        train_table = self.s.CASTable('trainset')
        max_objs = int(self.s.freq(train_table, inputs='_nObjects_').Frequency['NumVar'].max())

        targets = ['_nObjects_']
        for i in range(0, max_objs):
            targets.append('_Object%d_' % i)
            for sp in ["xmin", "ymin", "xmax", "ymax"]:
                targets.append('_Object%d_%s' % (i, sp))

        inputVars = []
        inputVars.insert(0, '_image_')

        solver = MomentumSolver(learning_rate=0.001, clip_grad_max=100, clip_grad_min=-100)
        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1,
                              reg_l2=0.005, freeze_layers_to='roi_pooling')
        data_specs = [DataSpec(type_='IMAGE', layer='data', data=inputVars),
                      DataSpec(type_='OBJECTDETECTION', layer='rois', data=targets)]
        faster_rcnn_model = Toy_FasterRCNN(self.s, n_channels=3, n_classes=6, coord_type='Coco')
        res_gpu = faster_rcnn_model.fit(train_table,
                                    optimizer=optimizer,
                                    data_specs=data_specs,
                                    n_threads=2,
                                    record_seed=13309,
                                    force_equal_padding=True)

        self.assertEqual(res_gpu.status_code, 0)
        self.assertEqual(len(res_gpu.OptIterHistory.Loss), 1)
        self.assertEqual(len(res_gpu.messages), 31)

        optimizer = Optimizer(algorithm=solver, mini_batch_size=4, log_level=2, max_epochs=1, reg_l2=0.005,
                              freeze_layers_to='fastrcnn')

        res = faster_rcnn_model.fit(train_table,
                       optimizer=optimizer,
                       data_specs=data_specs,
                       n_threads=2,
                       record_seed=13309,
                       force_equal_padding=True)
        self.assertEqual(res.status_code, 0)
        self.assertEqual(len(res.messages), 31)
        self.assertEqual(len(res.OptIterHistory.Loss), 1)


if __name__ == '__main__':
    unittest.main()

