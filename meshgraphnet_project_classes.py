
import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Multiply

class FiniteElementMesh3D():
    '''
    The class of which a specific Mesh will be a member.
    The term "Mesh" is quite general so the name used here is more specific.
    This project is for experience and a 2D mesh will be adequate for that 
    purpose. The 2D mesh class will be a subclass of this class.

    A 3D mesh is comprised of vertices, edges connecting those vertices, 
    faces comprised of closed loops of edges, and elements comprised of 
    closed volumes bounded by faces.

    Assumptions include: 
    * Abaqus .inp files, .stp and/or .step files will be read using a TBD 
    library.
    
    '''
    
    def __init__(self, vertices = [], edges = [], faces = [], elements = []):
        ''' 
        FiniteElementMesh constructor can be called with no arguments, 
        resulting in an empty mesh. Loading and saving can be done
        subsequently, using FiniteElementMesh methods.
        '''
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.elements = elements
        self._build_edge_lookup()
        self._build_edge_properties()

    def load(self, filename):
        # Load mesh from file; not yet authored
        self._build_edge_vertex_lookup()
        self._build_edge_properties()
        pass

    def save(self, filename):
        pass

    def _build_edge_lookup(self):
        self.edge_lookup = [[] for _ in self.vertices]
        for edge in self.edges:
            self.edge_lookup[edge.properties['vertex'][0]].append(edge.properties['id'])
            self.edge_lookup[edge.properties['vertex'][1]].append(edge.properties['id'])

    def _build_edge_properties(self):
        for edge in self.edges:
            edge.properties['vector'] = \
                  self.vertices[edge.properties['vertex'][1]].properties['position'] \
                - self.vertices[edge.properties['vertex'][0]].properties['position']
            edge.properties['length'] = \
                math.sqrt(sum([edge.properties['vector'][position]**2 for position in len(edge.properties['vector'])]))


class FiniteElementMesh2D(FiniteElementMesh3D):
    '''
    The class of which a specific Mesh will be a member.
    The term "Mesh" is quite general so the name used here is more specific.

    A 2D mesh is comprised of vertices, edges connecting those vertices,
    and elements comprised of closed loops of edges. For consistency with 3D
    mesh definitions, the 2D elements will be called faces.

    Inherits load and save methods from FiniteElementMesh3D.

    There is nothing about this class that inherently limits it to representing
    a 2D mesh. There is no intent to initialize volumes, but since the vertex, 
    edge and face properties are not defined within this class, they could 
    be defined in 3D and be non-co-planar. To that end, there is no particular
    need for this Class to exist.

    '''

    def __init__(self, vertices = [], edges = [], faces = []):
        ''' 
        FiniteElementMesh constructor can be called with no arguments, 
        resulting in an empty mesh. Loading and saving can be done
        subsequently, using FiniteElementMesh3D methods.
        '''
        self.vertices = vertices
        self.edges = edges
        self.faces = faces


class FiniteElementMesh1D(FiniteElementMesh3D):
    '''
    Placeholder for the class of which a specific Mesh will be a member.
    The term "Mesh" is quite general so the name used here is more specific.
    During this project, this placeholder will only possibly be authored.
    This project is for experience with the target being a 2D mesh, but if
    adequately frustrating problems emerge, there may be a time for trying
    a 1D mesh graph net. There is, however, no literature supporting the 1D
    use case for mesh graph nets.

    A 1D mesh is comprised of vertices and elements connecting those vertices.
    For consistency with 2D and 3D mesh definitions, the 1D elements will be 
    called edges.

    Inherits load and save methods from FiniteElementMesh3D.

    There is nothing about this class that inherently limits it to representing
    a 2D mesh. There is no intent to initialize volumes, but since the vertex, 
    edge and face properties are not defined within this class, they could 
    be defined in 3D and be non-co-planar. To that end, there is no particular
    need for this Class to exist.
    '''


class MeshVertex():
    '''
    Placeholder for the class used to define a vertex in a mesh.
    '''

    def __init__(self, property_dict):
        '''
        MeshVertex constructor is called with a property_dict; the property
        dictionary must be defined by each project.
        
        Minimum anticipated entries for this project:
        id                  (int)
        position            (2 element list float)
        temperature         (float)
        '''
        self.properties = property_dict
        

class MeshEdge():
    '''
    Placeholder for the class used to define an edge in a mesh.
    '''

    def __init__(self, property_dict):
        '''
        MeshEdge constructor is called with a property_dict; the property
        dictionary must be defined by each project.

        Minimum anticipated entries for this project:
        id                  (int)
        vertex              (2 element list int)
        '''
        self.properties = property_dict


class MeshFace():
    '''
    Placeholder for the class used to define a face in a mesh.
    '''

    def __init__(self, property_dict):
        '''
        MeshFace constructor is called with a property_dict; the property
        dictionary must be defined by each project.

        This structure is not anticipated to be used in this project.
        Minimum anticipated entries for a future project:
        id                  (int)
        face                (2+ element list int)
        material            (TBD)
        doc                 (float)
        '''
        self.properties = property_dict


class MeshElement():
    '''
    Placeholder for the class used to define an element in a mesh.
    '''

    def __init__(self, property_dict):
        '''
        MeshElement constructor is called with a property_dict; the property
        dictionary must be defined by each project.

        This structure is not anticipated to be used in this project.
        '''
        self.properties = property_dict


class MGNGraph():
    '''
    The class of which a specific graph will be a member.
    The term "Graph" is quite general so the name used here is more specific.
    There is no inherent physical dimension to a graph.

    '''

    def __init__(self, vertices = [], edges = []):
        ''' 
        MGNGraph constructor can be called with no arguments, resulting in an 
        empty graph. Loading and saving can be done subsequently.
        '''
        self.vertices = vertices
        self.edges = edges
        self._build_edge_lookup()

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def _build_edge_lookup(self):
        self.edge_lookup = [[] for _ in self.vertices]
        for edge in self.edges:
            self.edge_lookup[edge.properties['vertex'][0]].append(edge.properties['id'])
            self.edge_lookup[edge.properties['vertex'][1]].append(edge.properties['id'])

    def create_subgraph(cls, mesh, input_vertex_id, subgraph_depth):
        '''For a given vertex within a mesh, produce a graph of the 
        neighborhood out to a specified depth.'''
        # Define the initial data structures
        vertex_ids = [[input_vertex_id]]
        vertex_id_list = [input_vertex_id]
        vertices = [GraphVertex(mesh.vertices[input_vertex_id].properties)]
        edge_ids = [[]]
        edge_id_list = []
        edges = []

        # Loop, building up the subgraph
        for depth in range(subgraph_depth):
            # For each vertex in the current layer, find all new connecting edges
            for vertex_id in vertex_id[depth]:
                for edge_id in mesh.edge_lookup[vertex_id]:
                    if (edge_id not in edge_id_list):
                        if (len(edge_ids) == depth - 1):
                            edge_ids.append([edge_id])
                        else:
                            edge_ids[depth].append(edge_id)
                        edge_id_list.append(edge_id)
                        edges.append(GraphEdge(mesh.edges[edge_id].properties))
            # For each edge in the new layer, find all vertices that aren't already in the graph
            for edge_id in edge_ids[depth]:
                for vertex_id in mesh.edges[edge_id].properties['vertex']:
                    if (vertex_id not in vertex_id_list):
                        if (len(vertex_ids) == depth + 1):
                            vertex_ids.append([vertex_id])
                        else:
                            vertex_ids[depth+1].append(vertex_id)
                        vertex_id_list.append(vertex_id)
                        vertices.append(GraphVertex(mesh.vertices[vertex_id].properties))

        # Build and return the MGNGraph
        return MGNGraph(vertices, edges)
    
    def update_subgraph(self, mesh):
        '''
        This may not be necessary. The subgraphs are graphs consisting of 
        GraphVertex and GraphEdge instances, defined using the properties of
        MeshVertex and MeshEdge instances but without deep copying. Thus the 
        subgraph properties may be automatically updating to/with/from the mesh
        properties. 
        Written in case it is necessary.
        I don't know which is more efficient: updating the properties of each 
        subgraph (which is redundant as each vertex is in many subgraphs) or
        creating new subgraphs every epoch. The answer depends on the depth
        and mesh connectivity.
        '''
        for vertex in self.vertices:
            vertex.properties = mesh.vertices[vertex.properties['id']].properties
        for edge in self.edges:
            edge.properties = mesh.edges[edge.properties['id']].properties


class GraphVertex():
    '''
    Placeholder docstring for the class used to define a vertex in a graph.
    Creates an inputs dictionary that is used by the MGN model
    '''

    def __init__(self, property_dict):
        '''
        GraphVertex constructor is called with a property_dict; the property
        dictionary must be defined by each project.
        
        Automatically one-hot encodes boolean entries in property_dict.

        Minimum anticipated entries for this project:
        id                  (int)
        position            (2 element list float)
        temperature         (float)
        boundary            (boolean)
        boundary_value      (float)

        Then it will gain other properties later including:
        encoding            (encoding_size list float)
        current_latent      (TBD list float, either encoding_size or encoding_size * (1 to message_passing_depth))
        decoded_temperature (float)
        '''
        self.properties = property_dict
        self.inputs = self._build_inputs()

    def _build_inputs(self):
        inputs = {}
        for property in self.properties:
            if (property in ['id']):
                pass
            else:
                if (type(property) in [np.float32, float]):
                    inputs[property] = self.properties[property]
                if (type(property) in [list]):
                    if (type(self.properties[property][0]) in [np.float32, float]):
                        for index in len(self.properties[property]):
                            inputs[property + str(index)] = self.properties[property][index]
                if (type(property) == bool):
                    inputs[property + str(0)] = 1. * (1 - self.properties[property])
                    inputs[property + str(1)] = 1. * self.properties[property]
        return inputs
        

class GraphEdge():
    '''
    Placeholder docstring for the class used to define an edge in a graph.
    Creates an inputs dictionary that is used by the MGN model
    '''

    def __init__(self, property_dict, vertex1, vertex2):
        '''
        MeshEdge constructor is called with a property_dict; the property
        dictionary must be defined by each project.

        Minimum anticipated entries for this project:
        id                  (int)
        vertex              (2 element list int)

        Built inputs include:
        vector              (2 element list float)
        length              (float)

        Then it will gain other properties later including:
        encoding            (encoding_size list float)
        current_latent      (TBD list float, either encoding_size or encoding_size * (1 to message_passing_depth))
        '''
        self.properties = property_dict
        self.inputs = self._build_inputs(vertex1, vertex2)

    def _build_inputs(self, vertex1, vertex2):
        inputs = {}
        for property in self.properties:
            if (property in ['id']):
                pass
            elif (property in ['vertex']):
                displacement_vector = vertex2.properties['position'] - vertex1.properties['position']
                if (type(displacement_vector) in [np.float32, float]):
                    edge_length = abs(displacement_vector)
                    inputs['vector'] = displacement_vector
                else:
                    edge_length = math.sqrt(sum([displacement_vector[component] for component in range(len(displacement_vector))]))
                    for component in len(displacement_vector):
                        inputs['vector' + str(component)] = displacement_vector[component]
                inputs['length'] = edge_length
            else:
                if (type(property) in [np.float32, float]):
                    inputs[property] = self.properties[property]
                if (type(property) in [list]):
                    if (type(self.properties[property][0]) in [np.float32, float]):
                        for index in len(self.properties[property]):
                            inputs[property + str(index)] = self.properties[property][index]
                if (type(property) == bool):
                    inputs[property + str(0)] = 1. * (1 - self.properties[property])
                    inputs[property + str(1)] = 1. * self.properties[property]
        return inputs


class MeshGraphNet():
    '''
    Class for a Mesh Graph Network.
    While this class is generalizable, it is written for the current project
    meaning that it assumes the MGN is learning to predict the temperature
    evolution of a system, e.g. updating one variable only.
    '''

    initializer = 'glorot_uniform'
    primary_activation = 'tanh'
    final_activation = 'linear'

    def __init__(self, graph_mesh_definition):
        '''
        The constructor for MeshGraphNet requires a definition dict including:
        mesh: the mesh for this instance
        depth: the number of times to pass messages
        width: the dimensionality of the latent space
        decoded_variables: a list of names of the variables output from the MGN
        '''
        self.mesh = graph_mesh_definition['mesh']
        self.latent_space_length = graph_mesh_definition['width']
        self.message_passing_depth = graph_mesh_definition['depth']
        self.decoded_variables = graph_mesh_definition['decoded_variables']  
        self.subgraphs = []
        for vertex in self.mesh.vertices:
            self.subgraphs.append(MGNGraph.create_subgraph(self.mesh, vertex.properties['id'], self.message_passing_depth))
        self.vertex_encoder = MeshGraphNet._define_encoder(self.subgraphs[0].vertices[0], self.latent_space_length)
        self.edge_encoder =   MeshGraphNet._define_encoder(self.subgraphs[0].edges[0],    self.latent_space_length)
        self.processor_edge = MeshGraphNet._define_message_passing_edge(self.latent_space_length)
        self.processor_vertex = MeshGraphNet._define_message_passing_vertex(self.latent_space_length)
        self.vertex_decoder = MeshGraphNet._define_decoder(self.latent_space_length, len(self.decoded_variables))
        self.model = self.build_model(self.message_passing_depth)

    def _define_encoder(cls, graph_object, dimensions):
        return Dense(dimensions, input_dim = len(graph_object.inputs), activation = MeshGraphNet.primary_activation, kernel_initializer = MeshGraphNet.initializer, bias_initializer = MeshGraphNet.initializer)

    def _define_message_passsing_edge(cls, dimensions):
        return Dense(dimensions, input_dim = 3 * dimensions, activation = MeshGraphNet.primary_activation, kernel_initializer = MeshGraphNet.initializer, bias_initializer = MeshGraphNet.initializer)

    def _define_message_passsing_vertex(cls, dimensions):
        #return Dense(dimensions, input_dim = len(graph_object.inputs), activation = MeshGraphNet.primary_activation, kernel_initializer = MeshGraphNet.initializer, bias_initializer = MeshGraphNet.initializer)

    def _define_decoder(cls, dimensions, output_dimensions):
        return Dense(output_dimensions, input_dim = dimensions, activation = MeshGraphNet.final_activation, kernel_initializer = MeshGraphNet.initializer, bias_initializer = MeshGraphNet.initializer)

    def _define_layer_norm(cls, dimensions):
        #return

    def _build_model(message_passing_depth):
        model = Sequential()
        model.add
        