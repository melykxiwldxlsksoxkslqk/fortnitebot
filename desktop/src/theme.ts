import { createTheme, MantineColorsTuple } from '@mantine/core'

// Кастомный фиолетовый акцент
const epicPurple: MantineColorsTuple = [
  '#f5f0ff',
  '#e6dcff',
  '#ccb5ff',
  '#b08eff',
  '#9a6dff',
  '#8b5cf6', // основной
  '#7c4ddb',
  '#6d3fc0',
  '#5e31a5',
  '#4f248a',
]

// Кастомный cyan для выделений
const epicCyan: MantineColorsTuple = [
  '#e0fcff',
  '#b8f3ff',
  '#8ceaff',
  '#5fe0ff',
  '#33d6ff',
  '#06b6d4', // основной
  '#0596b0',
  '#04768c',
  '#035668',
  '#023644',
]

export const theme = createTheme({
  primaryColor: 'epicPurple',
  colors: {
    epicPurple,
    epicCyan,
    dark: [
      '#C1C2C5',
      '#A6A7AB',
      '#909296',
      '#5C5F66',
      '#373A40',
      '#2C2E33',
      '#25262B',
      '#1A1B1E',
      '#141517',
      '#101113',
    ],
  },
  
  fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif',
  fontFamilyMonospace: 'JetBrains Mono, Consolas, Monaco, monospace',
  
  headings: {
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif',
    fontWeight: '600',
  },
  
  radius: {
    xs: '4px',
    sm: '6px',
    md: '8px',
    lg: '12px',
    xl: '16px',
  },
  
  shadows: {
    xs: '0 1px 2px rgba(0, 0, 0, 0.05)',
    sm: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  },
  
  components: {
    Button: {
      defaultProps: {
        radius: 'md',
      },
      styles: {
        root: {
          fontWeight: 500,
          transition: 'all 0.2s ease',
        },
      },
    },
    
    Card: {
      defaultProps: {
        radius: 'lg',
        shadow: 'sm',
      },
      styles: {
        root: {
          backgroundColor: 'var(--mantine-color-dark-7)',
          border: '1px solid var(--mantine-color-dark-5)',
        },
      },
    },
    
    TextInput: {
      defaultProps: {
        radius: 'md',
      },
      styles: {
        input: {
          backgroundColor: 'var(--mantine-color-dark-7)',
          borderColor: 'var(--mantine-color-dark-5)',
          '&:focus': {
            borderColor: 'var(--mantine-color-epicPurple-5)',
          },
        },
      },
    },
    
    Textarea: {
      defaultProps: {
        radius: 'md',
      },
      styles: {
        input: {
          backgroundColor: 'var(--mantine-color-dark-7)',
          borderColor: 'var(--mantine-color-dark-5)',
        },
      },
    },
    
    Select: {
      defaultProps: {
        radius: 'md',
      },
    },
    
    NumberInput: {
      defaultProps: {
        radius: 'md',
      },
    },
    
    Modal: {
      defaultProps: {
        radius: 'lg',
        overlayProps: {
          blur: 4,
        },
      },
    },
    
    Table: {
      styles: {
        table: {
          backgroundColor: 'transparent',
        },
        th: {
          backgroundColor: 'var(--mantine-color-dark-7)',
          borderColor: 'var(--mantine-color-dark-5)',
          fontWeight: 600,
        },
        td: {
          borderColor: 'var(--mantine-color-dark-5)',
        },
        tr: {
          '&:hover': {
            backgroundColor: 'var(--mantine-color-dark-6)',
          },
        },
      },
    },
    
    NavLink: {
      styles: {
        root: {
          borderRadius: 'var(--mantine-radius-md)',
          marginBottom: '4px',
          transition: 'all 0.15s ease',
          '&[data-active]': {
            backgroundColor: 'var(--mantine-color-epicPurple-9)',
            '&:hover': {
              backgroundColor: 'var(--mantine-color-epicPurple-8)',
            },
          },
        },
      },
    },
    
    Badge: {
      defaultProps: {
        radius: 'md',
      },
    },
    
    Alert: {
      defaultProps: {
        radius: 'md',
      },
    },
    
    Paper: {
      defaultProps: {
        radius: 'md',
      },
      styles: {
        root: {
          backgroundColor: 'var(--mantine-color-dark-7)',
          border: '1px solid var(--mantine-color-dark-5)',
        },
      },
    },
    
    Tooltip: {
      defaultProps: {
        radius: 'md',
        withArrow: true,
      },
    },
    
    Switch: {
      styles: {
        track: {
          cursor: 'pointer',
        },
      },
    },
  },
})
