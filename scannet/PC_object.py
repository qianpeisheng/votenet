# The object class represents an object in a scene. 
# Objects constitute the memory bank. 

class PC_object:
    def __init__(self, object_info) -> None:
        self.scene_name = object_info['scene_name']
        self.object_id = int(object_info['object_id'])
        self.object_class = int(object_info['object_class'])
        self._is_in_memory_bank = False # prepend with _ to avoid name conflict with the property name

    def __str__(self) -> str:
        return f'scene_name: {self.scene_name}, object_id: {self.object_id}, object_class: {self.object_class}, \
        is_in_memory_bank: {self.is_in_memory_bank}'
    
    def add_to_memory_bank(self):
        self._is_in_memory_bank = True
    
    def remove_from_memory_bank(self):
        self._is_in_memory_bank = False

    @property
    def is_in_memory_bank(self):
        return self._is_in_memory_bank