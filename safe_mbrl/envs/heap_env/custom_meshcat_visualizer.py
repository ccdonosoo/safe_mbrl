import meshcat
import meshcat.geometry as g
import meshcat.animation
import pinocchio as pin
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
import os

def isMesh(geometry_object):
    """Check whether the geometry object contains a Mesh supported by MeshCat"""
    if geometry_object.meshPath == "":
        return False

    _, file_extension = os.path.splitext(geometry_object.meshPath)
    if file_extension.lower() in [".dae", ".obj", ".stl"]:
        return True

    return False

class AnimeMeshcatVisualizer(MeshcatVisualizer):
    """ A custom visualizer that extends the MeshcatVisualizer class from Pinocchio to support animations. """

    def display(self, q=None, animation_frame=None):
        """Display the robot at configuration q in the viewer by placing all the bodies."""
        if q is not None:
            pin.forwardKinematics(self.model, self.data, q)

        if self.display_collisions:
            self.updatePlacements(pin.GeometryType.COLLISION, animation_frame)

        if self.display_visuals:
            self.updatePlacements(pin.GeometryType.VISUAL, animation_frame)

        if self.display_frames:
            self.updateFrames(animation_frame)
    
    def updatePlacements(self, geometry_type, animation_frame=None):
        if geometry_type == pin.GeometryType.VISUAL:
            geom_model = self.visual_model
            geom_data = self.visual_data
        else:
            geom_model = self.collision_model
            geom_data = self.collision_data

        pin.updateGeometryPlacements(self.model, self.data, geom_model, geom_data)
        for visual in geom_model.geometryObjects:
            visual_name = self.getViewerNodeName(visual, geometry_type)
            # Get mesh pose.
            M = geom_data.oMg[geom_model.getGeometryId(visual.name)]
            # Manage scaling: force scaling even if this should be normally handled by MeshCat (but there is a bug here)
            if isMesh(visual):
                scale = np.asarray(visual.meshScale).flatten()
                S = np.diag(np.concatenate((scale, [1.0])))
                T = np.array(M.homogeneous).dot(S)
            else:
                T = M.homogeneous
            if animation_frame is not None:
                animation_frame[visual_name].set_transform(T)
                continue
            # Update viewer configuration.
            self.viewer[visual_name].set_transform(T)

        for visual in self.static_objects:
            visual_name = self.getViewerNodeName(visual, pin.GeometryType.VISUAL)
            M = visual.placement
            T = M.homogeneous
            if animation_frame is not None:
                animation_frame[visual_name].set_transform(T)
                continue
            self.viewer[visual_name].set_transform(T)
    
    def updateFrames(self, animation_frame=None):
        """
        Updates the frame visualizations with the latest transforms from model data.
        """
        pin.updateFramePlacements(self.model, self.data)
        for fid in self.frame_ids:
            frame_name = self.model.frames[fid].name
            frame_viz_name = "%s/%s" % (self.viewerFramesGroupName, frame_name)
            if animation_frame is not None:
                animation_frame[frame_viz_name].set_transform(
                    self.data.oMf[fid].homogeneous
                )
                continue
            self.viewer[frame_viz_name].set_transform(
                self.data.oMf[fid].homogeneous
            )