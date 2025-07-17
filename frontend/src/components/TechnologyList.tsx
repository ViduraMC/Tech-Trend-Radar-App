import React, { useState } from 'react';

interface Technology {
  id: number;
  name: string;
  category: string;
  description: string;
  keywords: string[];
  first_detected: string;
  last_updated: string;
}

interface TechnologyListProps {
  technologies: Technology[];
}

const TechnologyList: React.FC<TechnologyListProps> = ({ technologies }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  // Get unique categories
  const categories = ['all', ...Array.from(new Set(technologies.map(tech => tech.category)))];

  // Filter technologies
  const filteredTechnologies = technologies.filter(tech => {
    const matchesSearch = tech.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         tech.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         tech.keywords.some(keyword => keyword.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesCategory = selectedCategory === 'all' || tech.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      'AI/ML': 'bg-purple-100 text-purple-800',
      'Blockchain': 'bg-yellow-100 text-yellow-800',
      'Cloud': 'bg-blue-100 text-blue-800',
      'Frontend': 'bg-green-100 text-green-800',
      'Backend': 'bg-red-100 text-red-800',
      'Mobile': 'bg-indigo-100 text-indigo-800',
      'Data': 'bg-pink-100 text-pink-800',
      'DevOps': 'bg-gray-100 text-gray-800',
      'IoT': 'bg-orange-100 text-orange-800',
      'Quantum': 'bg-cyan-100 text-cyan-800',
      'AR/VR': 'bg-emerald-100 text-emerald-800',
      'Cybersecurity': 'bg-rose-100 text-rose-800'
    };
    return colors[category] || 'bg-gray-100 text-gray-800';
  };

  if (technologies.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-400 text-6xl mb-4">🔍</div>
        <p className="text-gray-500">No technologies found</p>
        <p className="text-sm text-gray-400">Technologies will appear here once added to the database</p>
      </div>
    );
  }

  return (
    <div>
      {/* Filters */}
      <div className="mb-6 flex flex-col sm:flex-row gap-4">
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search technologies..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <div className="sm:w-48">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {categories.map(category => (
              <option key={category} value={category}>
                {category === 'all' ? 'All Categories' : category}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Results count */}
      <div className="mb-4">
        <p className="text-sm text-gray-600">
          Showing {filteredTechnologies.length} of {technologies.length} technologies
        </p>
      </div>

      {/* Technologies Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Technology
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Category
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Description
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Keywords
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                First Detected
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Last Updated
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredTechnologies.map((tech) => (
              <tr key={tech.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                      <span className="text-blue-600 font-semibold text-sm">
                        {tech.name.charAt(0).toUpperCase()}
                      </span>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-gray-900">{tech.name}</div>
                      <div className="text-sm text-gray-500">ID: {tech.id}</div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getCategoryColor(tech.category)}`}>
                    {tech.category}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <div className="text-sm text-gray-900 max-w-xs truncate">
                    {tech.description}
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="flex flex-wrap gap-1">
                    {tech.keywords.slice(0, 3).map((keyword, index) => (
                      <span
                        key={index}
                        className="inline-flex px-2 py-1 text-xs bg-gray-100 text-gray-800 rounded"
                      >
                        {keyword}
                      </span>
                    ))}
                    {tech.keywords.length > 3 && (
                      <span className="inline-flex px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded">
                        +{tech.keywords.length - 3}
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {new Date(tech.first_detected).toLocaleDateString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {new Date(tech.last_updated).toLocaleDateString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {filteredTechnologies.length === 0 && (
        <div className="text-center py-8">
          <p className="text-gray-500">No technologies match your search criteria</p>
        </div>
      )}
    </div>
  );
};

export default TechnologyList; 