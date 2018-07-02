'''
Copyright 2018 IN3PD

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
class RecEngine(object):

    def __init__(self, items_list, items_index, users_to_item, items_to_user):
        self.items_list = items_list
        self.items_index = items_index
        self.users_to_item = users_to_item
        self.items_to_user = items_to_user
        self.n_items = len(items_list)
        self.n_users = len(items_to_user)


    # Trains the model
    def train(self, test_playlists):
        """
        @return: the model itself
        @rtype: RecEngine
        """
        return self
    '''
    def convert_test(self, test_playlists):
        converted_test_playlists={}
        for key in test_playlists:
            converted_test_playlists[key]=[self.track_indexes[j] for j in test_playlists[key]]
        return converted_test_playlists
    '''
    def get_recommendation(self, pid):
        raise NotImplementedError
