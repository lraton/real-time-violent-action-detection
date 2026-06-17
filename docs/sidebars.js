// @ts-check

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Setup & Installation',
      items: ['getting-started'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'System Design & Pipeline',
      items: ['architecture', 'preprocessing'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'AI Models & Training',
      items: ['training'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Performance & Thesis',
      items: ['evaluation', 'thesis'],
      collapsed: false,
    },
  ],
};

export default sidebars;
