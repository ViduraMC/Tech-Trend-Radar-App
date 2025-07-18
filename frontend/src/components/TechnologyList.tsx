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
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [sortBy, setSortBy] = useState<'name' | 'category' | 'date'>('name');

  // Handle null/undefined technologies
  if (!technologies || !Array.isArray(technologies)) {
    return (
      <div style={{ textAlign: 'center', padding: '2rem' }}>
        <div style={{ fontSize: '4rem', color: '#9ca3af', marginBottom: '1rem' }}>⏳</div>
        <p style={{ color: '#6b7280', fontSize: '1.125rem', marginBottom: '0.5rem' }}>Loading technologies...</p>
        <p style={{ color: '#9ca3af', fontSize: '0.875rem' }}>Please wait while we fetch the data</p>
      </div>
    );
  }

  // Get unique categories
  const categories = ['all', ...Array.from(new Set(technologies.map(tech => tech.category)))];

  // Filter and sort technologies
  const filteredTechnologies = technologies
    .filter(tech => {
      const matchesSearch = (tech.name && tech.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
                           (tech.description && tech.description.toLowerCase().includes(searchTerm.toLowerCase())) ||
                           (tech.keywords && tech.keywords.some(keyword => keyword.toLowerCase().includes(searchTerm.toLowerCase())));
      const matchesCategory = selectedCategory === 'all' || tech.category === selectedCategory;
      return matchesSearch && matchesCategory;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return (a.name || '').localeCompare(b.name || '');
        case 'category':
          return (a.category || '').localeCompare(b.category || '');
        case 'date':
          return new Date(b.first_detected || 0).getTime() - new Date(a.first_detected || 0).getTime();
        default:
          return 0;
      }
    });

  const getCategoryColor = (category: string) => {
    const colors: Record<string, { bg: string; text: string; border: string }> = {
      'AI/ML': { bg: '#f3e8ff', text: '#7c3aed', border: '#c084fc' },
      'Blockchain': { bg: '#fef3c7', text: '#d97706', border: '#fbbf24' },
      'Cloud': { bg: '#dbeafe', text: '#2563eb', border: '#60a5fa' },
      'Frontend': { bg: '#dcfce7', text: '#16a34a', border: '#4ade80' },
      'Backend': { bg: '#fee2e2', text: '#dc2626', border: '#f87171' },
      'Mobile': { bg: '#e0e7ff', text: '#4f46e5', border: '#818cf8' },
      'Data': { bg: '#fce7f3', text: '#ec4899', border: '#f472b6' },
      'DevOps': { bg: '#f3f4f6', text: '#6b7280', border: '#9ca3af' },
      'IoT': { bg: '#fed7aa', text: '#ea580c', border: '#fb923c' },
      'Quantum': { bg: '#cffafe', text: '#0891b2', border: '#22d3ee' },
      'AR/VR': { bg: '#d1fae5', text: '#059669', border: '#34d399' },
      'Cybersecurity': { bg: '#fef2f2', text: '#dc2626', border: '#fca5a5' }
    };
    return colors[category] || { bg: '#f3f4f6', text: '#6b7280', border: '#9ca3af' };
  };

  if (technologies.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '2rem' }}>
        <div style={{ fontSize: '4rem', color: '#9ca3af', marginBottom: '1rem' }}>🔍</div>
        <p style={{ color: '#6b7280', fontSize: '1.125rem', marginBottom: '0.5rem' }}>No technologies found</p>
        <p style={{ color: '#9ca3af', fontSize: '0.875rem' }}>Technologies will appear here once added to the database</p>
      </div>
    );
  }

  return (
    <div>
      {/* Enhanced Filters */}
      <div style={{ 
        background: 'white', 
        borderRadius: '1rem', 
        padding: '1.5rem', 
        marginBottom: '1.5rem',
        boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
        border: '1px solid #e5e7eb'
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Search and Category Filter */}
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <div style={{ flex: '1', minWidth: '300px' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', fontWeight: '500', color: '#374151' }}>
                🔍 Search Technologies
              </label>
              <input
                type="text"
                placeholder="Search by name, description, or keywords..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.5rem',
                  fontSize: '0.875rem',
                  transition: 'all 0.2s'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = '#3b82f6';
                  e.target.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.1)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = '#d1d5db';
                  e.target.style.boxShadow = 'none';
                }}
              />
            </div>
            <div style={{ minWidth: '200px' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', fontWeight: '500', color: '#374151' }}>
                📂 Category Filter
              </label>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.5rem',
                  fontSize: '0.875rem',
                  backgroundColor: 'white'
                }}
              >
                {categories.map(category => (
                  <option key={category} value={category}>
                    {category === 'all' ? 'All Categories' : category}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* View Mode and Sort Controls */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <label style={{ fontSize: '0.875rem', fontWeight: '500', color: '#374151' }}>Sort by:</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                style={{
                  padding: '0.5rem',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.375rem',
                  fontSize: '0.875rem',
                  backgroundColor: 'white'
                }}
              >
                <option value="name">Name (A-Z)</option>
                <option value="category">Category</option>
                <option value="date">Date (Newest)</option>
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <button
                onClick={() => setViewMode('grid')}
                style={{
                  padding: '0.5rem',
                  border: viewMode === 'grid' ? '2px solid #3b82f6' : '1px solid #d1d5db',
                  borderRadius: '0.375rem',
                  background: viewMode === 'grid' ? '#eff6ff' : 'white',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                📱
              </button>
              <button
                onClick={() => setViewMode('list')}
                style={{
                  padding: '0.5rem',
                  border: viewMode === 'list' ? '2px solid #3b82f6' : '1px solid #d1d5db',
                  borderRadius: '0.375rem',
                  background: viewMode === 'list' ? '#eff6ff' : 'white',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                📋
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Results Summary */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '1rem',
        padding: '0 0.5rem'
      }}>
        <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>
          Showing {filteredTechnologies.length} of {technologies.length} technologies
        </p>
        {searchTerm && (
          <p style={{ fontSize: '0.875rem', color: '#3b82f6' }}>
            Search results for "{searchTerm}"
          </p>
        )}
      </div>

      {/* Technologies Display */}
      {viewMode === 'grid' ? (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', 
          gap: '1.5rem' 
        }}>
          {filteredTechnologies.map((tech) => {
            const categoryColors = getCategoryColor(tech.category || 'Uncategorized');
            return (
              <div
                key={tech.id || `tech-${Math.random()}`}
                style={{
                  background: 'white',
                  borderRadius: '1rem',
                  padding: '1.5rem',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
                  border: '1px solid #e5e7eb',
                  transition: 'all 0.2s',
                  cursor: 'pointer'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.1)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.05)';
                }}
              >
                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{
                      width: '3rem',
                      height: '3rem',
                      background: `linear-gradient(135deg, ${categoryColors.bg}, ${categoryColors.bg}dd)`,
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: `2px solid ${categoryColors.border}`
                    }}>
                      <span style={{ 
                        fontSize: '1.25rem', 
                        fontWeight: 'bold',
                        color: categoryColors.text
                      }}>
                        {(tech.name && tech.name.charAt(0)) || '?'}
                      </span>
                    </div>
                    <div>
                      <h3 style={{ 
                        fontSize: '1.25rem', 
                        fontWeight: 'bold', 
                        color: '#111827',
                        margin: '0 0 0.25rem 0'
                      }}>
                        {tech.name || 'Unnamed Technology'}
                      </h3>
                      <span style={{
                        background: categoryColors.bg,
                        color: categoryColors.text,
                        padding: '0.25rem 0.75rem',
                        borderRadius: '1rem',
                        fontSize: '0.75rem',
                        fontWeight: '600',
                        border: `1px solid ${categoryColors.border}`
                      }}>
                        {tech.category || 'Uncategorized'}
                      </span>
                    </div>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>
                    ID: {tech.id || 'N/A'}
                  </div>
                </div>

                {/* Description */}
                <p style={{ 
                  fontSize: '0.875rem', 
                  color: '#374151', 
                  lineHeight: '1.5',
                  marginBottom: '1rem'
                }}>
                  {tech.description && tech.description.length > 120 
                    ? `${tech.description.substring(0, 120)}...` 
                    : tech.description || 'No description available'
                  }
                </p>

                {/* Keywords */}
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {tech.keywords && tech.keywords.slice(0, 4).map((keyword, index) => (
                      <span
                        key={index}
                        style={{
                          background: '#f3f4f6',
                          color: '#6b7280',
                          padding: '0.25rem 0.5rem',
                          borderRadius: '0.375rem',
                          fontSize: '0.75rem',
                          fontWeight: '500'
                        }}
                      >
                        {keyword}
                      </span>
                    ))}
                    {tech.keywords && tech.keywords.length > 4 && (
                      <span style={{
                        background: '#f3f4f6',
                        color: '#9ca3af',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '0.375rem',
                        fontSize: '0.75rem'
                      }}>
                        +{tech.keywords.length - 4}
                      </span>
                    )}
                  </div>
                </div>

                {/* Footer */}
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  paddingTop: '1rem',
                  borderTop: '1px solid #f3f4f6',
                  fontSize: '0.75rem',
                  color: '#6b7280'
                }}>
                  <span>First: {tech.first_detected ? new Date(tech.first_detected).toLocaleDateString() : 'Unknown'}</span>
                  <span>Updated: {tech.last_updated ? new Date(tech.last_updated).toLocaleDateString() : 'Unknown'}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* List View */
        <div style={{ 
          background: 'white', 
          borderRadius: '1rem', 
          overflow: 'hidden',
          boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
          border: '1px solid #e5e7eb'
        }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f9fafb' }}>
                  <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                    Technology
                  </th>
                  <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                    Category
                  </th>
                  <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                    Description
                  </th>
                  <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                    Keywords
                  </th>
                  <th style={{ padding: '1rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '600', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                    First Detected
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredTechnologies.map((tech, index) => {
                  const categoryColors = getCategoryColor(tech.category || 'Uncategorized');
                  return (
                    <tr 
                      key={tech.id || `tech-${Math.random()}`} 
                      style={{ 
                        borderBottom: index < filteredTechnologies.length - 1 ? '1px solid #f3f4f6' : 'none',
                        transition: 'background-color 0.2s'
                      }}
                      onMouseOver={(e) => e.currentTarget.style.background = '#f9fafb'}
                      onMouseOut={(e) => e.currentTarget.style.background = 'white'}
                    >
                      <td style={{ padding: '1rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <div style={{
                            width: '2.5rem',
                            height: '2.5rem',
                            background: `linear-gradient(135deg, ${categoryColors.bg}, ${categoryColors.bg}dd)`,
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            border: `1px solid ${categoryColors.border}`
                          }}>
                            <span style={{ 
                              fontSize: '1rem', 
                              fontWeight: 'bold',
                              color: categoryColors.text
                            }}>
                              {(tech.name && tech.name.charAt(0)) || '?'}
                            </span>
                          </div>
                          <div>
                            <div style={{ fontSize: '0.875rem', fontWeight: '600', color: '#111827' }}>
                              {tech.name || 'Unnamed Technology'}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>
                              ID: {tech.id || 'N/A'}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td style={{ padding: '1rem' }}>
                        <span style={{
                          background: categoryColors.bg,
                          color: categoryColors.text,
                          padding: '0.25rem 0.75rem',
                          borderRadius: '1rem',
                          fontSize: '0.75rem',
                          fontWeight: '600',
                          border: `1px solid ${categoryColors.border}`
                        }}>
                          {tech.category || 'Uncategorized'}
                        </span>
                      </td>
                      <td style={{ padding: '1rem' }}>
                        <div style={{ fontSize: '0.875rem', color: '#374151', maxWidth: '300px' }}>
                          {tech.description && tech.description.length > 100 
                            ? `${tech.description.substring(0, 100)}...` 
                            : tech.description || 'No description available'
                          }
                        </div>
                      </td>
                      <td style={{ padding: '1rem' }}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                          {tech.keywords && tech.keywords.slice(0, 3).map((keyword, index) => (
                            <span
                              key={index}
                              style={{
                                background: '#f3f4f6',
                                color: '#6b7280',
                                padding: '0.125rem 0.375rem',
                                borderRadius: '0.25rem',
                                fontSize: '0.75rem'
                              }}
                            >
                              {keyword}
                            </span>
                          ))}
                          {tech.keywords && tech.keywords.length > 3 && (
                            <span style={{
                              background: '#f3f4f6',
                              color: '#9ca3af',
                              padding: '0.125rem 0.375rem',
                              borderRadius: '0.25rem',
                              fontSize: '0.75rem'
                            }}>
                              +{tech.keywords.length - 3}
                            </span>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: '1rem', fontSize: '0.875rem', color: '#6b7280' }}>
                        {tech.first_detected ? new Date(tech.first_detected).toLocaleDateString() : 'Unknown'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {filteredTechnologies.length === 0 && (
        <div style={{ 
          textAlign: 'center', 
          padding: '3rem',
          background: 'white',
          borderRadius: '1rem',
          boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
          border: '1px solid #e5e7eb'
        }}>
          <div style={{ fontSize: '4rem', color: '#9ca3af', marginBottom: '1rem' }}>🔍</div>
          <p style={{ color: '#6b7280', fontSize: '1.125rem', marginBottom: '0.5rem' }}>
            No technologies match your search criteria
          </p>
          <p style={{ color: '#9ca3af', fontSize: '0.875rem' }}>
            Try adjusting your search terms or category filter
          </p>
        </div>
      )}
    </div>
  );
};

export default TechnologyList; 